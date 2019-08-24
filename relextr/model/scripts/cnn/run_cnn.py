#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn

from torch.optim import Adagrad
from torch.autograd import Variable

from model import CNN
from data_loader import load_batches

def main():
    train_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5/train.vectors')
    valid_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5/valid.vectors')
    test_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5/test.vectors')

    mapping = {
        'no_relation': 0,
        'in_relation': 1,
    }

    params = {
	'emb_dim': 3072, 
	'n_filters': 128, 
	'filter_sizes': (4, 8, 16, 32),
        'out_dim': 2, 
	'dropout': 0.5
    }
    
    model = CNN(**params)
    loss_function = nn.BCEWithLogitsLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_function.to(device)
  
    optimizer = Adagrad(model.parameters())
 
    # training 
    best_valid_loss = float('inf')
    for epoch in range(20):
        train_metrics = train(network, optimizer, loss_func, train_batches)
        print_metrics(train_metrics, 'Train')

        valid_metrics = evaluate(network, valid_batches, loss_func)
        print_metrics(valid_metrics, 'Valid')

        valid_loss = valid_metrics['loss']
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(network.state_dict(), 'semrel.model_5.pt')

    test_metrics = evaluate(network, test_batches, loss_func)
    print_metrics(test_metrics, 'Test')


def print_metrics(metrics, prefix):
    print(f'{prefix} - Loss: {metrics["loss"]}, '
          f'Accuracy: {metrics["accuracy"]}, '
          f'Precision: {metrics["precision"]}, '
          f'Recall: {metrics["recall"]}, '
          f'Fscore: {metrics["fscore"]}')


if __name__ == "__main__":
    main()
