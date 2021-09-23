#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 20:29:46 2021

@author: Gong Dongsheng
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from metrics import compute_accuracy
from model import GraphIsomorphismNetwork
from preprocess import load_data, separate_data


# Hyper parameters
batch_size = 32
iters_per_epoch = 50
n_epochs = 350
lr = 0.01
seed = 0
fold_idx = 0
dataset = "COLLAB"
num_layers = 5
num_mlp_layers = 2
hidden_dim = 64
final_dropout = 0.5
graph_pooling_type = "sum"
neighbor_pooling_type = "sum"
learn_eps = False
degree_as_tag = True
_device = "cuda"
device = torch.device(_device)

# fixed random seed
torch.manual_seed(seed)
np.random.seed(seed)
if _device == "cuda":
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# load data
graphs, num_classes = load_data(dataset, degree_as_tag)
train_graphs, test_graphs = separate_data(graphs, seed, fold_idx)

# initialize model
model = GraphIsomorphismNetwork(
    num_layers=num_layers,
    num_mlp_layers=num_mlp_layers,
    input_dim=train_graphs[0].node_features.shape[1],
    hidden_dim=hidden_dim,
    output_dim=num_classes,
    final_dropout=final_dropout,
    learn_eps=learn_eps,
    neighbor_pooling_type=neighbor_pooling_type,
    graph_pooling_type=graph_pooling_type,
    device=device
)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
loss_func = nn.CrossEntropyLoss()

# start training
acc_train_ls, acc_test_ls = [], []
for epoch in range(1, n_epochs + 1):
    # start training
    total_loss = 0.0
    model.train()
    for i in range(iters_per_epoch):
        selected_idx = np.random.permutation(len(train_graphs))[:batch_size]
        graphs = [train_graphs[idx] for idx in selected_idx]
        output = model(graphs)
        labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
        loss = loss_func(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
        
    model.eval()
    acc_train = compute_accuracy(model, train_graphs, device)
    acc_test = compute_accuracy(model, test_graphs, device)
    acc_train_ls.append(acc_train)
    acc_test_ls.append(acc_test)
    print("Epoch {} | Train ACC {} % | Test ACC {} %".format(
        epoch,
        np.round(acc_train * 100, 6),
        np.round(acc_test * 100, 6)
    ))


plt.figure(figsize=(10, 8))
plt.title("Dataset: {}".format(dataset))
plt.ylim(0.4, 1)
plt.plot(range(len(acc_train_ls)), acc_train_ls, label="train", linewidth=2)
plt.plot(range(len(acc_test_ls)), acc_test_ls, label="test", linewidth=2)
plt.xlabel("n epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()