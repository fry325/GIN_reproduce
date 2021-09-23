#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 21:06:28 2021

@author: Gong Dongsheng
"""

import torch
import numpy as np


def compute_accuracy(model, graphs, device, batch_size=64):
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), batch_size):
        curr_idx = idx[i: i + batch_size]
        if len(curr_idx) == 0:
            continue
        output.append(model([graphs[j] for j in curr_idx]).detach())
    output = torch.cat(output, 0)
    
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc = correct / float(len(graphs))
    
    return acc