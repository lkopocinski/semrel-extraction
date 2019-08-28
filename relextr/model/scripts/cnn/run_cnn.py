#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn

from torch.optim import Adagrad
from torch.autograd import Variable

from model import CNN
from data_loader import load_batches
from train import train_model, evaluate_model

def main():
    train_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5_temp/train_.vectors')
    valid_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5_temp/valid_.vectors')
    test_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5_temp/test_.vectors')

    labels2idx = {
        'no_relation': 0,
        'in_relation': 1,
    }

    params = {
	'emb_dim': 3072, 
	'n_filters': 128, 
	'filter_sizes': (4, 8, 16, 32),
#	'filter_sizes': (4, 8),
        'out_dim': 2, 
	'dropout': 0.5
    }
    
    network = CNN(**params)
    loss_function = nn.BCEWithLogitsLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    loss_function.to(device)
  
    optimizer = Adagrad(network.parameters())
    print(network) 
    # training 
    best_valid_loss = float('inf')
    for epoch in range(20):
        train_metrics = train_model(
            network, train_batches, loss_function, optimizer, labels2idx, device
        )
        print_metrics(train_metrics, 'Train')
        
        valid_metrics = evaluate_model(
            network, valid_batches, loss_function, labels2idx, device
        )
        print_metrics(valid_metrics, 'Valid')

        valid_loss = valid_metrics['loss']
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(network.state_dict(), 'semrel.model.cnn.pt')

    # testing
    test_metrics = evaluate_model(
        network, test_batches, loss_function, labels2idx, device
    )
    print_metrics(test_metrics, 'Test')


def print_metrics(metrics, prefix):
    print(f'{prefix} - Loss: {metrics["loss"]}, '
          f'Accuracy: {metrics["accuracy"]}, '
          f'Precision: {metrics["precision"]}, '
          f'Recall: {metrics["recall"]}, '
          f'Fscore: {metrics["fscore"]}')


if __name__ == "__main__":
    main()
