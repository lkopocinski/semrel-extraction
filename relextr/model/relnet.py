#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelNet(nn.Module):
    """
        in_dim: konkatenacja wektorow (300), hidden_dim: warstwa ukryta (300),
        out_dim: liczba relacji
    """
    def __init__(self, in_dim=3072, hidden_dim=512, out_dim=2, dropout=0.5):
        super(RelNet, self).__init__()
        self.f1 = nn.Linear(in_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ x: a concatenation of word vectors """
        h = self.f1(x)
        o = self.f2(h)
        return o

    def extract_layer_weights(self, layername):
        layer = self.__dict__['_modules'][layername]
        return layer.weight.data.numpy()

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
