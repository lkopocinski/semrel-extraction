#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import argcomplete
import mlflow
import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from relnet import RelNet
from relextr.model.scripts.utils.batches import Dataset, Sampler
from utils.metrics import Metrics, save_metrics
from utils.utils import is_better_fscore, parse_config, get_device


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data-in', required=True, type=str, help="Directory with train, validation and test dataset.")
    parser.add_argument('--config', required=True, type=str, help="Config file path")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    device = get_device()
    args = get_args(argv)
    config = parse_config(Path(args.config))
    init_mlflow(config)

    keys = Dataset.load_keys(Path(config['keys']))
    dataset = Dataset(config['methods'], keys)

    train_batch_gen = DataLoader(dataset, batch_size=config['batch_size'], sampler=Sampler(dataset, 'train'), num_workers=8)
    valid_batch_gen = DataLoader(dataset, batch_size=config['batch_size'], sampler=Sampler(dataset, 'valid'), num_workers=8)
    test_batch_gen = DataLoader(dataset, batch_size=config['batch_size'], sampler=Sampler(dataset, 'test'), num_workers=8)

    network = RelNet(in_dim=dataset.vector_size)
    network.to(device)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    # Log learning params
    mlflow.log_params({
        'batch_size': config['batch_size'],
        'train_set_size': len(train_batch_gen.sampler),
        'valid_set_size': len(valid_batch_gen.sampler),
        'test_set_size': len(test_batch_gen.sampler),
        'vector_size': dataset.vector_size,
        'epochs': config['epochs'],
        'optimizer': optimizer.__class__.__name__,
        'loss_function': loss_func.__class__.__name__
    })

    best_valid_fscore = (0.0, 0.0)

    for epoch in range(config['epochs']):
        print(f'\nEpoch: {epoch} / {config["epochs"]}')

        # Train
        train_metrics = train(network, optimizer, loss_func, train_batch_gen, device)
        print(f'Train:\n{train_metrics}')
        log_metrics(train_metrics, 'train', epoch)

        # Validate
        valid_metrics = evaluate(network, valid_batch_gen, loss_func, device)
        print(f'Valid:\n{valid_metrics}')
        log_metrics(valid_metrics, 'valid', epoch)

        # Fscore stopping
        if is_better_fscore(valid_metrics.fscore, best_valid_fscore):
            best_valid_fscore = valid_metrics.fscore
            torch.save(network.state_dict(), config["model"]["name"])
            mlflow.log_artifact(f'./{config["model"]["name"]}')

    # Test
    test_metrics = test(RelNet(dataset.vector_size), config['model']['name'], test_batch_gen, loss_func, device)
    print(f'\n\nTest: {test_metrics}')
    save_metrics(test_metrics, 'metrics.txt')
    log_metrics(test_metrics, 'test')


def init_mlflow(config):
    conf = config['mlflow']
    mlflow.set_tracking_uri(conf['tracking_uri'])
    mlflow.set_experiment(conf['experiment_name'])
    mlflow.set_tags(conf['tags'])
    mlflow.set_tag('method', config['methods'])

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


def train(network, optimizer, loss_function, batches, device):
    metrics = Metrics()

    network.train()
    for data, labels in batches:
        optimizer.zero_grad()

        data = data.to(device)
        target = labels.to(device)

        output = network(data).squeeze(0)
        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()

        metrics.update(output.cpu(), target.cpu(), loss.item(), len(batches))

    return metrics


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


def test(network, model_path, batches, loss_function, device):
    network.load(model_path)
    network.to(device)
    return evaluate(network, batches, loss_function, device)


if __name__ == "__main__":
    main()
