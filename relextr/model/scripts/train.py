#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import reduce

import argcomplete
import mlflow
import torch
import torch.nn as nn
from metrics import Metrics, save_metrics
from relnet import RelNet
from torch.autograd import Variable
from torch.optim import Adagrad
from utils import load_batches, labels2idx

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Runing on: {device}.')


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', required=True, type=int, help="How many epochs should the model be trained.")
    parser.add_argument('-n', '--model_name', required=True, type=str, help="Save file name for a trained model.")
    parser.add_argument('-b', '--batch_size', required=True, type=int, help="Batch size.")
    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help="Directory with train, validation, test dataset.")
    parser.add_argument('-s', '--sent2vec', required=True, type=str, help="Sent2vec word embeddings model path.")

    argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


def init_mlflow(uri, experiment):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    print(f'-- mlflow --'
          f'\nserver: {mlflow.get_tracking_uri()}'
          f'\nexperiment: {experiment}')


def get_set_size(dataset_batches):
    return reduce((lambda x, y: x + len(y)), dataset_batches, 0)


def is_better_fscore(fscore, best_fscore):
    return fscore[0] > best_fscore[0] and fscore[1] > best_fscore[1]


def log_metrics(metrics, step, prefix):
    mlflow.log_metric(key=f'{prefix}_loss', value=metrics.loss, step=step)
    mlflow.log_metric(f'{prefix}_accuracy', metrics.accuracy, step=step)
    mlflow.log_metric(f'{prefix}_precision_pos', metrics.precision[1], step=step)
    mlflow.log_metric(f'{prefix}_precision_neg', metrics.precision[0], step=step)
    mlflow.log_metric(f'{prefix}_recall_pos', metrics.recall[1], step=step)
    mlflow.log_metric(f'{prefix}_recall_neg', metrics.recall[0], step=step)
    mlflow.log_metric(f'{prefix}_fscore_pos', metrics.fscore[1], step=step)
    mlflow.log_metric(f'{prefix}_fscore_neg', metrics.fscore[0], step=step)


def main(argv=None):
    init_mlflow(
        uri='http://0.0.0.0:5000',
        experiment='no_substituted_pairs_sent2vec'
    )

    args = get_args(argv)

    network = RelNet()
    network.to(device)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    train_batches = load_batches(f'{args.dataset_dir}/train.vectors', args.batch_size)
    valid_batches = load_batches(f'{args.dataset_dir}/valid.vectors', args.batch_size)
    test_batches = load_batches(f'{args.dataset_dir}/test.vectors', args.batch_size)

    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('train_set_size', get_set_size(train_batches))
    mlflow.log_param('valid_set_size', get_set_size(valid_batches))
    mlflow.log_param('test_set_size', get_set_size(test_batches))
    mlflow.log_param('epochs', args.epochs)

    best_valid_fscore = [0.0, 0.0]

    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch} / {args.epochs}')

        train_metrics = train(network, optimizer, loss_func, train_batches, device)
        print(f'Train:\n{train_metrics}')
        log_metrics(train_metrics, epoch, 'train')

        valid_metrics = evaluate(network, valid_batches, loss_func, device)
        print(f'Valid:\n{valid_metrics}')
        log_metrics(valid_metrics, epoch, 'valid')

        if is_better_fscore(valid_metrics.fscore, best_valid_fscore):
            best_valid_fscore = valid_metrics.fscore
            torch.save(network.state_dict(), args.model_name)
            mlflow.log_artifact(f'../{args.model_name}', '/artifacts/models/')

    # Test
    test_metrics = test(args.model_name, test_batches, loss_func, device)
    print(f'\n\nTest: {test_metrics}')
    save_metrics(test_metrics, 'metrics.txt')
    log_metrics(test_metrics, 0, 'test')


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


def test(model_path, batches, loss_function, device):
    network = RelNet()
    network.load(model_path)
    network.to(device)
    return evaluate(network, batches, loss_function, device)


if __name__ == "__main__":
    main()
