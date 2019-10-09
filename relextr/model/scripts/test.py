#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adagrad

from relnet import RelNet
from utils import load_batches, compute_accuracy, labels2idx, \
    compute_precision_recall_fscore, Metrics, save_metrics

try:
    import argcomplete
except ImportError:
    argcomplete = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Runing on: {device}.')


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', required=True, type=str, help="Load pre-trained model.")
    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help="Directory with test dataset.")

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    loss_func = nn.CrossEntropyLoss()

    test_batches = load_batches(f'{args.dataset_dir}/test.vectors', 20)

    network = RelNet(out_dim=2)
    network.load(args.model_name)
    network.to(device)

    test_metrics = evaluate(network, test_batches, loss_func, device)
    print(f'\n\nTest: {test_metrics}')
    save_metrics(test_metrics, 'metrics.txt')


def evaluate(network, batches, loss_function, device):
    metrics = Metrics()
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
            metrics.update(loss.item(), accuracy, precision, recall, fscore, len(batches))
            metrics.update_count(output.cpu(), target.cpu())

    return metrics


if __name__ == "__main__":
    main()
