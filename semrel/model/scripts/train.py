#!/usr/bin/env python3.6

from pathlib import Path

import click
import mlflow
import torch
import torch.nn as nn
from torch.optim import Adagrad, Optimizer
from torch.utils.data import DataLoader

from semrel.model import runs
from semrel.model.scripts import RelNet
from semrel.model.scripts.utils import parse_config, get_device, \
    is_better_loss, ignored
from semrel.model.scripts.utils.data_loader import get_loaders
from semrel.model.scripts.utils.metrics import Metrics


@click.command()
@click.option('--config', required=True,
              type=click.Path(exists=True),
              help='File with training params.')
def main(config):
    device = get_device()
    config = parse_config(Path(config))

    runs_name = config['learn_params']['runs']
    train_runs = runs.RUNS[runs_name]
    model_name = f'{runs_name}.pt'

    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])

    for index, params in train_runs.items():
        with ignored(Exception):
            with mlflow.start_run():
                print(f'\nRUN: {index} WITH: {params}')

                in_domain = params.get(runs.IN_DOMAIN_KEY)
                out_domain = params.get(runs.OUT_DOMAIN_KEY)
                lexical_split = params.get(runs.LEXICAL_SPLIT_KEY, False)
                methods = params.get(runs.METHODS_KEY, [])

                mlflow.set_tags({
                    'in_domain': in_domain,
                    'out_domain': out_domain,
                    'lexical_split': lexical_split,
                    'methods': ', '.join(methods),
                })

                loaders = get_loaders(
                    data_dir=config['dataset']['dir'],
                    keys_file=config['dataset']['keys'],
                    vectors_files=[
                        f'{method}.rel.pt' for method in methods
                    ],
                    batch_size=config['learn_params']['batch_size'],
                    balanced=True,
                    lexical_split=lexical_split,
                    in_domain=in_domain,
                    out_domain=out_domain
                )

                network = RelNet(
                    in_dim=loaders.vector_size,
                    **config['net_params']
                )
                network = network.to(device)
                optimizer = Adagrad(
                    network.parameters(),
                    lr=config['net_params']['learning_rate']
                )
                loss_func = nn.CrossEntropyLoss()

                # Log learning params
                mlflow.log_params({
                    'train size': len(loaders.train.sampler),
                    'valid size': len(loaders.valid.sampler),
                    'test size': len(loaders.test.sampler),
                    'vector size': loaders.vector_size,
                    'optimizer': optimizer.__class__.__name__,
                    'loss function': loss_func.__class__.__name__,
                    **config['learn_params'],
                    **config['net_params']
                })

                best_valid_loss = None

                print('Epochs:', end=" ")
                for epoch in range(config['learn_params']['epochs']):
                    print(epoch, end=" ")

                    # Train
                    train_metrics = train(
                        network, optimizer, loaders.train, loss_func, device
                    )
                    log_metrics(train_metrics, 'train', epoch)

                    # Validate
                    valid_metrics, _ = evaluate(
                        network, loaders.valid, loss_func, device
                    )
                    log_metrics(valid_metrics, 'valid', epoch)

                    # Loss stopping
                    if is_better_loss(valid_metrics.loss, best_valid_loss):
                        best_valid_loss = valid_metrics.loss
                        torch.save(network.state_dict(), model_name)
                        mlflow.log_artifact(f'./{model_name}')

                # Test
                test_network = RelNet(
                    in_dim=loaders.vector_size,
                    **config['net_params']
                )
                test_metrics = test(
                    test_network, model_name, loaders.test, loss_func, device
                )

                print(f'\n\nTest: {test_metrics}')
                log_metrics(test_metrics, 'test')


def log_metrics(metrics, prefix: str, step: int = 0):
    try:
        mlflow.log_metric(f'{prefix}_loss', metrics.loss)
    except AttributeError:
        pass

    mlflow.log_metrics({
        f'{prefix}_acc': metrics.accuracy,
        f'{prefix}_prec_pos': metrics.precision[1],
        f'{prefix}_prec_neg': metrics.precision[0],
        f'{prefix}_rec_pos': metrics.recall[1],
        f'{prefix}_rec_neg': metrics.recall[0],
        f'{prefix}_f_pos': metrics.fscore[1],
        f'{prefix}_f_neg': metrics.fscore[0]
    }, step=step)


def train(
        network: RelNet,
        optimizer: Optimizer,
        batches: DataLoader,
        loss_function,
        device: torch.device
) -> Metrics:
    metrics = Metrics()
    network.train()

    for data, labels in batches:
        optimizer.zero_grad()

        data = data.to(device)
        target = labels.to(device)

        output = network(data).squeeze(0)
        try:
            loss = loss_function(output, target)
        except IndexError:
            print(
                f'\nOutput: {output}'
                f'\nTarget: {target}'
            )
            continue

        loss.backward()
        optimizer.step()

        metrics.update(
            predicted=output.cpu(),
            targets=target.cpu(),
            loss=loss.item(),
            batches=len(batches)
        )

    return metrics


def evaluate(
        network: RelNet,
        batches: DataLoader,
        loss_function,
        device: torch.device
) -> Metrics:
    metrics = Metrics()
    network.eval()

    with torch.no_grad():
        for data, labels in batches:
            data = data.to(device)
            target = labels.to(device)

            output = network(data).squeeze(0)
            try:
                loss = loss_function(output, target)
            except IndexError:
                print(
                    f'\nOutput: {output}'
                    f'\nTarget: {target}'
                )
                continue

            metrics.update(
                predicted=output.cpu(),
                targets=target.cpu(),
                loss=loss.item(),
                batches=len(batches)
            )

    return metrics


def test(
        network: RelNet,
        model_path: str,
        batches: DataLoader,
        loss_function,
        device: torch.device
) -> Metrics:
    network.load(model_path)
    network.to(device)
    return evaluate(network, batches, loss_function, device)


if __name__ == "__main__":
    main()
