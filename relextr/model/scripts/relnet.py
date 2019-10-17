#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class RelNet(nn.Module):
    """
        in_dim: konkatenacja wektorow (1024), hidden_dim: warstwa ukryta (512),
        out_dim: liczba relacji
    """

    def __init__(self, in_dim=4096, hidden_dim=512, out_dim=2, dropout=0.5):
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


#class SyntacticRelNet(nn.Module):
#    """
#        in_dim: konkatenacja wektorow (1024), hidden_dim: warstwa ukryta (512),
#        out_dim: liczba relacji
#    """
#    # todo: change hidden_dim
#    def __init__(self, in_dim=1400, hidden_dim=512, out_dim=2, dropout=0.5):
#        super(SyntacticRelNet, self).__init__()
#	# todo: vocab size V
#	self.emb = nn.Embedding(V, 64)
#
#	self.l1 = nn.LSTM(64, 512)
#	self.l2 = nn.LSTM(64, 512)
#
#        self.f1 = nn.Linear(in_dim, hidden_dim)
#        self.f2 = nn.Linear(hidden_dim, hidden_dim)
#        self.f3 = nn.Linear(hidden_dim, out_dim)
#        self.dropout = nn.Dropout(dropout)
#
#    # how to prepare t1 and t2
#    def forward(self, x, t1, t2):
#        """ x: a concatenation of word vectors """
#
#	e1 = self.emb(t1)
#	e2 = self.emb(t2)
#
#	# todo: extract the output from last timestep
#	l1 = self.l1(e1)
#	l2 = self.l2(e2)
#
#	# todo: concatenate
#	h1 = self.f1(torch.concatenate([x, l1, l2]))
#	h2 = self.f2(h1)
#	o = self.f3(h2)
#
#        return o
#
#    def extract_layer_weights(self, layername):
#        layer = self.__dict__['_modules'][layername]
#        return layer.weight.data.numpy()
#
#    def load(self, model_path):
#        self.load_state_dict(torch.load(model_path))
#
#    def predict(self, data):
#        output = self(torch.FloatTensor([data]))
#        _, predicted = torch.max(output, dim=1)
#        return predicted.item()
#

