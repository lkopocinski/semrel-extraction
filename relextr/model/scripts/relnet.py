#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class RelNet(nn.Module):
    """
        in_dim: vectors concatenation size, hidden_dim: hidden layer dimension,
        out_dim: relations quantity
    """

    def __init__(self, in_dim=3448, hidden_dim=512, out_dim=2, dropout=0.5):
        super(RelNet, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ x: a concatenation of word vectors """
        h1 = self.f1(x)
        h2 = self.f2(h1)
        o = self.f3(h2)
        return o

    def extract_layer_weights(self, layername):
        layer = self.__dict__['_modules'][layername]
        return layer.weight.data.numpy()

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def predict(self, data):
        output = self(torch.FloatTensor([data]))
        _, predicted = torch.max(output, dim=1)
        return predicted.item()
