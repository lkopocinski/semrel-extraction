#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import argcomplete
import mlflow
import torch
import torch.nn as nn
from metrics import Metrics, save_metrics
from relnet import RelNet
from torch.autograd import Variable
from utils import load_batches, labels2idx, get_set_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Runing on: {device}.')


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model_name', required=True, type=str, help="Load pre-trained model.")
    parser.add_argument('-b', '--batch_size', required=True, type=int, help="Batch size.")
    parser.add_argument('-d', '--dataset_dir', required=True, type=str, help="Directory with test dataset.")
    parser.add_argument('-s', '--sent2vec', required=False, type=str, help="Sent2vec word embeddings model path.")
    parser.add_argument('-f', '--fasttext', required=False, type=str, help="Fasttext word embeddings model path.")

    argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


def init_mlflow(uri, experiment, tag):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    mlflow.set_tag(key=tag[0], value=tag[1])
    print(f'-- mlflow --'
          f'\nserver: {mlflow.get_tracking_uri()}'
          f'\nexperiment: {experiment}'
          f'\ntag: {tag[0]} - {tag[1]}')


def main(argv=None):
    init_mlflow(
        uri='http://0.0.0.0:5000',
        experiment='no_experiment',
        tag=('key', 'value')
    )

    args = get_args(argv)

    loss_func = nn.CrossEntropyLoss()

    test_batches = load_batches(f'{args.dataset_dir}/test.vectors_', args.batch_size)

    # Log learning params
    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('test_set_size', get_set_size(test_batches))

    network = RelNet()
    network.load(args.model_name)
    network.to(device)

    metrics = evaluate(network, test_batches, loss_func, device)
    print(f'\n\nTest: {metrics}')
    save_metrics(metrics, 'metrics.txt')

    mlflow.log_metrics({
        f'loss': metrics.loss,
        f'accuracy': metrics.accuracy,
        f'precision_pos': metrics.precision[1],
        f'precision_neg': metrics.precision[0],
        f'recall_pos': metrics.recall[1],
        f'recall_neg': metrics.recall[0],
        f'fscore_pos': metrics.fscore[1],
        f'fscore_neg': metrics.fscore[0]
    })


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

            metrics.update(output.cpu(), target.cpu(), loss.item(), len(batches))

    return metrics


if __name__ == "__main__":
    main()
