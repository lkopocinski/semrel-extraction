#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import argcomplete
import mlflow
import torch
import torch.nn as nn
from metrics import Metrics, save_metrics
from relnet import RelNet
from torch.autograd import Variable
from torch.optim import Adagrad
from .batches import BatchLoader
from utils import labels2idx, get_set_size, is_better_fscore

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Runing on: {device}.')


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, type=str, help="Directory with train, validation and test dataset.")
    parser.add_argument('--save-model-name', required=True, type=str, help="File name for a trained model.")
    parser.add_argument('--batch-size', required=True, type=int, help="Batch size.")
    parser.add_argument('--epochs', required=True, type=int, help="How many epochs should the model be trained on.")
    parser.add_argument('--vectorizer', required=False, type=str, choices={'sent2vec', 'fasttext', 'elmoconv'}, help="Vectorizer method")
    parser.add_argument('--vectors-model', required=False, type=str, help="Vectors model for vectorizer method path")

    argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


def main(argv=None):
    init_mlflow(
        uri='http://0.0.0.0:5001',
        experiment='no_experiment'
    )

    args = get_args(argv)

    network = RelNet()
    network.to(device)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    batch_loader = BatchLoader(args.batch_size)
    train_batches = batch_loader.load(f'{args.dataset_dir}/train.vectors')
    valid_batches = batch_loader.load(f'{args.dataset_dir}/valid.vectors')
    test_batches = batch_loader.load(f'{args.dataset_dir}/test.vectors')

    # Log learning params
    mlflow.log_params({
        'batch_size': args.batch_size,
        'train_set_size': get_set_size(train_batches),
        'valid_set_size': get_set_size(valid_batches),
        'test_set_size': get_set_size(test_batches),
        'epochs': args.epochs,
        'optimizer': optimizer.__class__.__name__,
        'loss_function': loss_func.__class__.__name__
    })

    best_valid_fscore = (0.0, 0.0)

    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch} / {args.epochs}')

        # Train
        train_metrics = train(network, optimizer, loss_func, train_batches, device)
        print(f'Train:\n{train_metrics}')
        log_metrics(train_metrics, epoch, 'train')

        # Validate
        valid_metrics = evaluate(network, valid_batches, loss_func, device)
        print(f'Valid:\n{valid_metrics}')
        log_metrics(valid_metrics, epoch, 'valid')

        # Fscore stopping
        if is_better_fscore(valid_metrics.fscore, best_valid_fscore):
            best_valid_fscore = valid_metrics.fscore
            torch.save(network.state_dict(), args.model_name)
            mlflow.log_artifact(f'../{args.model_name}', '/artifacts/models/')

    # Test
    test_metrics = test(args.model_name, test_batches, loss_func, device)
    print(f'\n\nTest: {test_metrics}')
    save_metrics(test_metrics, 'metrics.txt')
    log_metrics(test_metrics, 0, 'test')


def init_mlflow(uri, experiment):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    print(f'-- mlflow --'
          f'\nserver: {mlflow.get_tracking_uri()}'
          f'\nexperiment: {experiment}')


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


def test(model_path, batches, loss_function, device):
    network = RelNet()
    network.load(model_path)
    network.to(device)
    return evaluate(network, batches, loss_function, device)


if __name__ == "__main__":
    main()
