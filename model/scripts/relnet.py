#!/usr/bin/env python3.6

from collections import OrderedDict

import torch
import torch.nn as nn


class RelNet(nn.Module):
    """
        in_dim: vectors concatenation size,
        hidden_dim: hidden layer dimension,
        out_dim: relations quantity
    """

    def __init__(
            self,
            in_dim: int = 2048,
            hidden_dim: int = 512,
            out_dim: int = 2,
            dropout: float = 0.5
    ):
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

    def forward(self, x: torch.tensor):
        """ x: a concatenation of word vectors """
        return self.net(x)

    def extract_layer_weights(self, layer_name: str):
        layer = self.__dict__['_modules'][layer_name]
        return layer.weight.data.numpy()

    def load(self, model_path: str):
        state_dict = torch.load(model_path)
        try:
            self.load_state_dict(state_dict)
        except Exception:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
