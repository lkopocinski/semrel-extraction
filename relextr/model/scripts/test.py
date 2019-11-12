#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import argcomplete
import mlflow
import torch
import torch.nn as nn
from batches import BatchLoader
from metrics import Metrics, save_metrics
from relnet import RelNet
from torch.autograd import Variable
from utils import labels2idx, is_better_fscore

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Runing on: {device}.')


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, type=str, help="Directory with test dataset.")
    parser.add_argument('--model-name', required=True, type=str, help="Load pre-trained model.")
    parser.add_argument('--batch-size', required=True, type=int, help="Batch size.")
    parser.add_argument('--vectorizer', required=False, type=str, choices={'sent2vec', 'fasttext', 'elmoconv'},
                        help="Vectorizer method")
    parser.add_argument('--vectors-model', required=False, type=str, help="Vectors model for vectorizer method path.")
    parser.add_argument('--tracking-uri', required=True, type=str, help="Mlflow tracking server uri.")
    parser.add_argument('--experiment-name', required=True, type=str, help="Mlflow tracking experiment name.")

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
    args = get_args(argv)

    init_mlflow(args.tracking_uri, args.experiment_name, tag=('key', 'value'))

    batch_loader = BatchLoader(args.batch_size)
    test_set = batch_loader.load(f'{args.dataset_dir}/test.vectors_')

    network = RelNet(in_dim=test_set.vector_size)
    network.load(args.model_name)
    network.to(device)
    loss_func = nn.CrossEntropyLoss()

    # Log learning params
    mlflow.log_params({
        'batch_size': args.batch_size,
        'test_set_size': test_set.size,
        'vector_size': test_set.vector_size,
        'epochs': args.epochs,
        'loss_function': loss_func.__class__.__name__
    })

    metrics = evaluate(network, test_set.batches, loss_func, device)
    print(f'\n\nTest: {metrics}')
    save_metrics(metrics, 'metrics_test.txt')
    log_metrics(metrics)


def log_metrics(metrics):
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
