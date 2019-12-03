#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy

import argcomplete
import mlflow
import torch
import traceback
import torch.nn as nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from relextr.model.config import RUNS
from relextr.model.scripts.utils.batches import Dataset, Sampler
from relnet import RelNet
from utils.metrics import Metrics, save_metrics
from utils.utils import is_better_fscore, parse_config, get_device, is_better_loss


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help="Config file path")
    argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


def main(argv=None):
    device = get_device()
    args = get_args(argv)
    config = parse_config(args.config)
    init_mlflow(config['mlflow'])

    runs = RUNS[config['runs']]

    for nr, params in runs.items():
        try:
            with mlflow.start_run():
                print(f'\nRUN: {nr} WITH {params}')

                lexical_split = params['lexical_split']
                in_domain = params['in_domain'] if 'in_domain' in params.keys() else None
                out_domain = params['out_domain'] if 'out_domain' in params.keys() else None
                methods = params['methods']

                mlflow.set_tags({
                    'lexical_split': lexical_split,
                    'in_domain': in_domain,
                    'out_domain': out_domain,
                    'methods': ', '.join(methods),
                })

                dataset = Dataset(
                    vectors_models=[m + '.rel.pt' for m in methods],
                    keys=Dataset.load_keys(config['keys'])
                )

                sampler = Sampler(
                    dataset,
                    balanced=True,
                    lexical_split=lexical_split,
                    in_domain=in_domain,
                    out_domain=out_domain
                )

                sampler.set_type = 'train'
                sampler_train = copy.copy(sampler)
                sampler.set_type = 'valid'
                sampler_valid = copy.copy(sampler)
                sampler.set_type = 'test'
                sampler_test = copy.copy(sampler)

                train_batch_gen = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler_train,
                                             num_workers=8)
                valid_batch_gen = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler_valid,
                                             num_workers=8)
                test_batch_gen = DataLoader(dataset, batch_size=config['batch_size'], sampler=sampler_test,
                                            num_workers=8)

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
                best_valid_loss = 0.0

                print('Epochs:', end=" ")
                for epoch in range(config['epochs']):
                    # print(f'Epoch: {epoch} / {config["epochs"]}')
                    print(epoch, end=" ")

                    # Train
                    train_metrics = train(network, optimizer, loss_func, train_batch_gen, device)
                    # print(f'Train:\n{train_metrics}')
                    log_metrics(train_metrics, 'train', epoch)

                    # Validate
                    valid_metrics = evaluate(network, valid_batch_gen, loss_func, device)
                    # print(f'Valid:\n{valid_metrics}')
                    log_metrics(valid_metrics, 'valid', epoch)

                    # Loss stopping
                    if is_better_loss(valid_metrics.loss, best_valid_loss):
                        best_valid_loss = valid_metrics.loss
                        torch.save(network.state_dict(), config["model"]["name"])
                        mlflow.log_artifact(f'./{config["model"]["name"]}')

                # Test
                test_metrics = test(RelNet(dataset.vector_size), config['model']['name'], test_batch_gen, loss_func,
                                    device)
                print(f'\n\nTest: {test_metrics}')
                save_metrics(test_metrics, 'metrics.txt')
                log_metrics(test_metrics, 'test')

        except Exception as e:
            print(f"\nIn {nr}'th run exception occurred", e)
            traceback.print_tb(e.__traceback__)
            continue


def init_mlflow(config):
    mlflow.set_tracking_uri(config['tracking_uri'])
    mlflow.set_experiment(config['experiment_name'])
    # mlflow.set_tags(config['tags'])

    print(f'\n-- mlflow --'
          f'\nserver: {mlflow.get_tracking_uri()}'
          f'\nexperiment: {config["experiment_name"]}'
          f'\ntag: {config["tags"]}')


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
    #    import pudb; pudb.set_trace()
    network.load(model_path)
    network.to(device)
    return evaluate(network, batches, loss_function, device)


if __name__ == "__main__":
    main()
