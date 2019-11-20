#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import argcomplete
import mlflow
import torch
import yaml
import torch.nn as nn
from torch.autograd import Variable

from relnet import RelNet
from utils.batches import BatchLoader
from utils.engines import VectorizerFactory
from utils.metrics import Metrics, save_metrics
from utils.utils import labels2idx

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Runing on: {device}.')


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, type=str, help="Directory with test dataset.")
    parser.add_argument('--config', required=True, type=str, help="Config file path.")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    device = get_device()
    args = get_args(argv)
    config = parse_config(args.config)
    init_mlflow(config['mlflow'])

    vectorizers = [VectorizerFactory.get_vectorizer(vectorizer['type'], vectorizer['model']) for vectorizer in config['vectorizers']]

    batch_loader = BatchLoader(config['batch_size'], vectorizers)
    test_set = batch_loader.load(f'{args.data_in}/test.vectors')

    network = RelNet(in_dim=test_set.vector_size)
    network.load(config['model']['name'])
    network.to(device)
    loss_func = nn.CrossEntropyLoss()

    # Log learning params
    mlflow.log_params({
        'batch_size': config['batch_size'],
        'test_set_size': test_set.size,
        'vector_size': test_set.vector_size,
        'loss_function': loss_func.__class__.__name__
    })

    metrics = evaluate(network, test_set.batches, loss_func, device)
    print(f'\n\nTest: {metrics}')
    save_metrics(metrics, 'metrics_test.txt')
    log_metrics(metrics)


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Runing on: {device}.')
    return device


def parse_config(path):
    with open(path, 'r', encoding='utf-8') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def init_mlflow(config):
    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    mlflow.set_tags(config['tags'])

    print(f'\n-- mlflow --'
          f'\nserver: {mlflow.get_tracking_uri()}'
          f'\nexperiment: {config["experiment_name"]}'
          f'\ntag: {config["tags"]}')


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
