#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import click
import torch
import torch.nn as nn
from torch.optim import Adagrad

import mlflow
from data_loader import get_loaders
from model import RUNS
from relnet import RelNet
from utils.metrics import Metrics
from utils.utils import parse_config, get_device, is_better_loss, ignored

logger = logging.getLogger(__name__)


@click.command()
@click.option('--config',
              required=True,
              type=click.Path(exists=True),
              help='File with training params.')
def main(config):
    device = get_device()
    config = parse_config(config)

    runs = RUNS[config['runs']]
    model_name = f'{config["runs"]}.pt'

    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])

    for idx, params in runs.items():
        with ignored(Exception):
            with mlflow.start_run():
                logger.info(f'\nRUN: {idx} WITH {params}')

                in_domain = params.get('in_domain')
                out_domain = params.get('out_domain')
                lexical_split = params.get('lexical_split', False)
                methods = params.get('methods', [])

                mlflow.set_tags({
                    'in_domain': in_domain,
                    'out_domain': out_domain,
                    'lexical_split': lexical_split,
                    'methods': ', '.join(methods),
                })

                train_loader, valid_loader, test_loader, vector_size = get_loaders(
                    data_dir=config['dataset']['dir'],
                    keys_file=config['dataset']['keys'],
                    vectors_files=[f'{m}.rel.pt' for m in methods],
                    batch_size=config['batch_size'],
                    balanced=True,
                    lexical_split=lexical_split,
                    in_domain=in_domain,
                    out_domain=out_domain
                )

                network = RelNet(in_dim=vector_size, **config['net_params'])
                network.to(device)
                optimizer = Adagrad(network.parameters(), lr=0.001)
                loss_func = nn.CrossEntropyLoss()

                # Log learning params
                mlflow.log_params({
                    'train size': len(train_loader),
                    'valid size': len(valid_loader),
                    'test size': len(test_loader),
                    'vector size': vector_size,
                    'optimizer': optimizer.__class__.__name__,
                    'loss function': loss_func.__class__.__name__,
                    **config['learn_params'],
                    **config['net_params']
                })

                best_valid_loss = None

                logger.info('Epochs:', end=" ")
                for epoch in range(config['epochs']):
                    logger.info(epoch, end=" ")

                    # Train
                    train_metrics = train(network, optimizer, loss_func, train_loader, device)
                    log_metrics(train_metrics, 'train', epoch)

                    # Validate
                    valid_metrics = evaluate(network, valid_loader, loss_func, device)
                    log_metrics(valid_metrics, 'valid', epoch)

                    # Loss stopping
                    if is_better_loss(valid_metrics.loss, best_valid_loss):
                        best_valid_loss = valid_metrics.loss
                        torch.save(network.state_dict(), model_name)
                        mlflow.log_artifact(f'./{model_name}')

                # Test
                test_network = RelNet(in_dim=vector_size, **config['net_params'])
                test_metrics = test(test_network, model_name, test_loader, loss_func, device)
                logger.info(f'\n\nTest: {test_metrics}')
                log_metrics(test_metrics, 'test')


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
