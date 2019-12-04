#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from collections import OrderedDict


class RelNet(nn.Module):
    """
        in_dim: vectors concatenation size,
        hidden_dim: hidden layer dimension,
        out_dim: relations quantity
    """

    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2, dropout=0.5):
        super(RelNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        """ x: a concatenation of word vectors """
        return self.net(x)

    def extract_layer_weights(self, layername):
        layer = self.__dict__['_modules'][layername]
        return layer.weight.data.numpy()

    def load(self, model_path):
        state_dict = torch.load(model_path)
        try:
            self.load_state_dict(state_dict)
        except Exception:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)

    def predict(self, data):
        output = self(torch.FloatTensor([data]))
        _, predicted = torch.max(output, dim=1)
        return predicted.item()
