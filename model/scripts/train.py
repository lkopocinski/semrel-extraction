#!/usr/bin/env python3.6

import logging
from pathlib import Path

import click
import mlflow
import torch
import torch.nn as nn
from torch.optim import Adagrad, Optimizer
from torch.utils.data import DataLoader

from model.scripts.utils.data_loader import get_loaders
from model.runs import RUNS
from model.scripts.relnet import RelNet
from model.scripts.utils.metrics import Metrics
from model.scripts.utils.utils import parse_config, get_device, is_better_loss, ignored

logger = logging.getLogger(__name__)


@click.command()
@click.option('--config', required=True,
              type=click.Path(exists=True),
              help='File with training params.')
def main(config):
    device = get_device()
    config = parse_config(Path(config))
    
    runs_name = config['learn_params']['runs']
    runs = RUNS[runs_name]
    model_name = f'{runs_name}.pt'

    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])

    for index, params in runs.items():
        with ignored(Exception):
            with mlflow.start_run():
                logger.info(f'\nRUN: {index} WITH: {params}')

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
                    vectors_files=[f'{method}.rel.pt' for method in methods],
                    batch_size=config['learn_params']['batch_size'],
                    balanced=True,
                    lexical_split=lexical_split,
                    in_domain=in_domain,
                    out_domain=out_domain
                )

                network = RelNet(in_dim=vector_size, **config['net_params'])
                network = network.to(device)
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
                    train_metrics = train(network, optimizer, train_loader, loss_func, device)
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


def log_metrics(metrics: Metrics, prefix: str, step: int = 0):
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


def train(network: RelNet, optimizer: Optimizer, batches: DataLoader, loss_function, device: torch.device):
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


def evaluate(network: RelNet, batches: DataLoader, loss_function, device: torch.device):
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


def test(network: RelNet, model_path: str, batches: DataLoader, loss_function, device: torch.device):
    network.load(model_path)
    network.to(device)
    return evaluate(network, batches, loss_function, device)


if __name__ == "__main__":
    main()
