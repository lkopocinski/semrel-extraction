#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import argcomplete
import mlflow
import torch
import torch.nn as nn
import yaml
from torch.autograd import Variable
from torch.optim import Adagrad

from relnet import RelNet
from utils.batches import BatchLoader
from utils.engines import VectorizerFactory
from utils.metrics import Metrics, save_metrics
from utils.utils import labels2idx, is_better_fscore


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, type=str, help="Directory with train, validation and test dataset.")
    parser.add_argument('--config', required=True, type=str, help="File name for a trained model.")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    device = get_device()
    args = get_args(argv)
    config = parse_config(args.config)
    init_mlflow(config['mlflow'])

    engine = VectorizerFactory.get_vectorizer(
        format=config['vectorizer']['type'],
        model_path=config['vectorizer']['model']
    )

    batch_loader = BatchLoader(config['batch_size'], engine)
    train_set = batch_loader.load(f'{args.data_in}/train.vectors')
    valid_set = batch_loader.load(f'{args.data_in}/valid.vectors')
    test_set = batch_loader.load(f'{args.data_in}/test.vectors')

    network = RelNet(in_dim=train_set.vector_size)
    network.to(device)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    # Log learning params
    mlflow.log_params({
        'batch_size': config['batch_size'],
        'train_set_size': train_set.size,
        'valid_set_size': valid_set.size,
        'test_set_size': test_set.size,
        'vector_size': train_set.vector_size,
        'epochs': config['epochs'],
        'optimizer': optimizer.__class__.__name__,
        'loss_function': loss_func.__class__.__name__
    })

    best_valid_fscore = (0.0, 0.0)

    for epoch in range(config['epochs']):
        print(f'\nEpoch: {epoch} / {config["epochs"]}')

        # Train
        train_metrics = train(network, optimizer, loss_func, train_set.batches, device)
        print(f'Train:\n{train_metrics}')
        log_metrics(train_metrics, epoch, 'train')

        # Validate
        valid_metrics = evaluate(network, valid_set.batches, loss_func, device)
        print(f'Valid:\n{valid_metrics}')
        log_metrics(valid_metrics, epoch, 'valid')

        # Fscore stopping
        if is_better_fscore(valid_metrics.fscore, best_valid_fscore):
            best_valid_fscore = valid_metrics.fscore
            torch.save(network.state_dict(), config["model"]["name"])
            mlflow.log_artifact(f'./{config["model"]["name"]}')

    # Test
    test_metrics = test(config['model']['name'], test_set.batches, test_set.vector_size, loss_func, device)
    print(f'\n\nTest: {test_metrics}')
    save_metrics(test_metrics, 'metrics.txt')
    log_metrics(test_metrics, 0, 'test')


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


def log_metrics(metrics, step, prefix):
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
    for batch in batches:
        optimizer.zero_grad()

        labels, data = zip(*batch)
        target = Variable(torch.LongTensor(labels2idx(labels))).to(device)
        data = torch.FloatTensor([data])

        output = network(data.to(device)).squeeze(0)
        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()

        metrics.update(output.cpu(), target.cpu(), loss.item(), len(batches))

    return metrics


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


def test(model_path, batches, in_dim, loss_function, device):
    network = RelNet(in_dim)
    network.load(model_path)
    network.to(device)
    return evaluate(network, batches, loss_function, device)


if __name__ == "__main__":
    main()
