#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adagrad

from relextr.model.scripts import RelNet
from relextr.model.scripts.utils import load_batches, print_metrics, compute_accuracy, labels2idx, compute_precision_recall_fscore

EPOCHS_QUANTITY = 30
MODEL_NAME = 'model'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Runing on: {device}.')


def main():
    network = RelNet(out_dim=2)
    network.to(device)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    train_batches = load_batches('relextr/model/datasets/train.vectors')
    valid_batches = load_batches('relextr/model/datasets/valid.vectors')
    test_batches = load_batches('relextr/model/datasets/test.vectors')

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS_QUANTITY):
        print(f'\nEpoch: {epoch} / {EPOCHS_QUANTITY}')

        train_metrics = train(network, optimizer, loss_func, train_batches, device)
        print_metrics(train_metrics, 'Train')

        valid_metrics = evaluate(network, valid_batches, loss_func, device)
        print_metrics(valid_metrics, 'Valid')

        valid_loss = valid_metrics['loss']
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(network.state_dict(), MODEL_NAME)

    test_metrics = evaluate(network, test_batches, loss_func, device)
    print_metrics(test_metrics, '\n\n-- Test --')


def train(network, optimizer, loss_func, batches, device):
    ep_loss, ep_acc, ep_prec, ep_rec, ep_f = 0.0, 0.0, 0.0, 0.0, 0.0
    network.train()

    for batch in batches:
        optimizer.zero_grad()

        labels, data = zip(*batch)
        target = Variable(torch.LongTensor(labels2idx(labels))).to(device)
        data = torch.FloatTensor([data])

        output = network(data.to(device)).squeeze(0)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(output, target)
        precision, recall, fscore = compute_precision_recall_fscore(output.cpu(), target.cpu())
        ep_loss += loss.item()

        ep_acc += accuracy
        ep_prec += precision
        ep_rec += recall
        ep_f += fscore

    return {
        'loss': ep_loss / len(batches),
        'accuracy': ep_acc / len(batches),
        'precision': ep_prec / len(batches),
        'recall': ep_rec / len(batches),
        'fscore': ep_f / len(batches)
    }


def evaluate(network, batches, loss_function, device):
    eval_loss, eval_acc, eval_prec, eval_rec, eval_f = 0.0, 0.0, 0.0, 0.0, 0.0
    network.eval()

    with torch.no_grad():
        for batch in batches:
            labels, data = zip(*batch)
            target = Variable(torch.LongTensor(labels2idx(labels))).to(device)
            data = torch.FloatTensor([data])

            output = network(data.to(device)).squeeze(0)
            loss = loss_function(output, target)

            accuracy = compute_accuracy(output, target)
            precision, recall, fscore = compute_precision_recall_fscore(output.cpu(), target.cpu())
            eval_loss += loss.item()

            eval_acc += accuracy
            eval_prec += precision
            eval_rec += recall
            eval_f += fscore

    return {
        'loss': eval_loss / len(batches),
        'accuracy': eval_acc / len(batches),
        'precision': eval_prec / len(batches),
        'recall': eval_rec / len(batches),
        'fscore': eval_f / len(batches)
    }


if __name__ == "__main__":
    main()
