#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import argcomplete
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from relnet import RelNet
from utils.batches import Dataset
from utils.metrics import Metrics, save_metrics
from utils.utils import parse_config, get_device

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
    init_mlflow(config)

    vectorizers = config['vectorizers']

    test_set = Dataset.from_file(Path(f'{args.data_in}/{config["dataset"]}/test.vectors'), vectorizers)
    test_batch_gen = DataLoader(test_set, batch_size=config['batch_size'], num_workers=8)

    network = RelNet(in_dim=test_set.vector_size)
    network.load(config['model']['name'])
    network.to(device)
    loss_func = nn.CrossEntropyLoss()

    # Log learning params
    mlflow.log_params({
        'batch_size': config['batch_size'],
        'test_set_size': len(test_set),
        'vector_size': test_set.vector_size,
        'loss_function': loss_func.__class__.__name__
    })

    metrics = evaluate(network, test_batch_gen, loss_func, device)
    print(f'\n\nTest: {metrics}')
    save_metrics(metrics, 'metrics_test.txt')
    log_metrics(metrics, 'test')


def init_mlflow(config):
    conf = config['mlflow']
    mlflow.set_tracking_uri(conf['tracking_uri'])
    mlflow.set_experiment(conf['experiment_name'])
    mlflow.set_tags(conf['tags'])
    mlflow.set_tag('method', config['vectorizers'])

    print(f'\n-- mlflow --'
          f'\nserver: {mlflow.get_tracking_uri()}'
          f'\nexperiment: {conf["experiment_name"]}'
          f'\ntag: {conf["tags"]}')


def log_metrics(metrics, prefix, step=0):
    mlflow.log_metrics({
        f'{prefix}_loss': metrics.loss,
        f'{prefix}_accuracy': metrics.accuracy,
        f'{prefix}_precision_pos': metrics.precision[1],
        f'{prefix}_precision_neg': metrics.precision[0],
        f'{prefix}_recall_pos': metrics.recall[1],
        f'{prefix}_recall_neg': metrics.recall[0],
        f'{prefix}_fscore_pos': metrics.fscore[1],
        f'{prefix}_fscore_neg': metrics.fscore[0]
    }, step=step)


def evaluate(network, batches, loss_function, device):
    metrics = Metrics()
    network.eval()

    with torch.no_grad():
        for data, labels in batches:
            data = data.to(device)
            target = labels.to(device)

            output = network(data).squeeze(0)
            loss = loss_function(output, target)

            metrics.update(output.cpu(), target.cpu(), loss.item(), len(batches))

    return metrics


if __name__ == "__main__":
    main()
