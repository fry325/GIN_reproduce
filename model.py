#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原作者的GIN稍作修改
url: https://github.com/weihua916/powerful-gnns
"""

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        if num_layers == 1:
            self.linear_layer = nn.Linear(input_dim, output_dim)
        else:
            self.first_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            if num_layers > 2:
                self.middle_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU()
                    )
                    for _ in range(num_layers - 2)
                ])
            self.last_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if self.num_layers == 1:
            x = self.linear_layer(x)
        else:
            x = self.first_layer(x)
            if self.num_layers > 2:
                for layer in self.middle_layers:
                    x = layer(x)
            x = self.last_layer(x)
        return x
    
    
class GraphIsomorphismNetwork(nn.Module):
    def __init__(
            self,
            num_layers,
            num_mlp_layers,
            input_dim,
            hidden_dim,
            output_dim,
            final_dropout,
            learn_eps,
            neighbor_pooling_type,
            graph_pooling_type,
            device
        ):
        super(GraphIsomorphismNetwork, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.neighbor_pooling_type = neighbor_pooling_type
        self.graph_pooling_type = graph_pooling_type
        self.device = device
        self.final_dropout = final_dropout
        
        self.eps = nn.Parameter(torch.zeros(num_layers - 1))
        self.mlp_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            self.mlp_layers.append(
                MLP(
                    num_mlp_layers,
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_dim
                )
            )
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        
        self.linear_predictions = nn.ModuleList()
        for i in range(num_layers):
            self.linear_predictions.append(
                nn.Linear(
                    input_dim if i == 0 else hidden_dim,
                    output_dim
                )
            )
    
    def _preprocess_neighbors_sumavgpool(self, graphs):
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(graphs):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        adj_idx = torch.cat(edge_mat_list, 1)
        adj_val = torch.ones(adj_idx.shape[1])
        num_nodes = start_idx[-1]
        
        if not self.learn_eps:
            self_loop_idx = torch.LongTensor([range(num_nodes), range(num_nodes)])
            self_loop_val = torch.ones(num_nodes)
            adj_idx = torch.cat((adj_idx, self_loop_idx), 1)
            adj_val = torch.cat((adj_val, self_loop_val), 0)
        
        adj = torch.sparse.FloatTensor(adj_idx, adj_val, torch.Size([num_nodes, num_nodes]))
        return adj.to(self.device)
        
    def _preprocess_graph_pool(self, graphs):
        '''
        或许你把graph pool矩阵画出来就清晰了
        其实就是一个B*N矩阵，B是batch size数，也是当前batch有多少graph，N是总结点数
        设输出图信号是h (N*d)，那么graph_pool * h: (B*d)每一行代表每个graph的所有图信号经过sum pooling
        '''
        start_idx = [0]
        for i, graph in enumerate(graphs):
            start_idx.append(start_idx[i] + len(graph.g))
        
        idx, elem = [], []
        for i, graph in enumerate(graphs):
            elem.extend([1 if self.graph_pooling_type=="sum" else 1.0/len(graph.g)]*len(graph.g))
            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1])])
        idx = torch.LongTensor(idx).transpose(0, 1)
        elem = torch.FloatTensor(elem)
        
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(graphs), start_idx[-1]]))
        return graph_pool.to(self.device)
    
    def next_layer(self, h, curr_layer, adj):
        pooled = torch.spmm(adj, h)
        
        if self.neighbor_pooling_type == 'average':
            degree = torch.spmm(adj, torch.ones((adj.shape[0], 1)).to(self.device))
            pooled = pooled / degree
            
        if self.learn_eps:
            pooled = pooled + (1 + self.eps[curr_layer]) * h
            
        pooled_rep = self.mlp_layers[curr_layer](pooled)
        h = self.bn_layers[curr_layer](pooled_rep)
        h = F.relu(h)
        return h
    
    def forward(self, graphs):
        x = torch.cat([graph.node_features for graph in graphs], 0).to(self.device)
        graph_pool = self._preprocess_graph_pool(graphs)
        adj = self._preprocess_neighbors_sumavgpool(graphs)
        
        hidden_representations = [x]
        h = x
        for curr_layer in range(self.num_layers - 1):
            h = self.next_layer(h, curr_layer, adj)
            hidden_representations.append(h)
        
        output = 0
        for curr_layer, h in enumerate(hidden_representations):
            pooled_h = torch.spmm(graph_pool, h)
            output += F.dropout(
                self.linear_predictions[curr_layer](pooled_h),
                self.final_dropout,
                training=self.training
            )
        return output