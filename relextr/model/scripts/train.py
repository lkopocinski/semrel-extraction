#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
from relnet import RelNet
from torch.autograd import Variable
from torch.optim import Adagrad
from utils import load_batches, labels2idx, \
    Metrics, save_metrics

from functools import reduce
import mlflow
import sent2vec

try:
    import argcomplete
except ImportError:
    argcomplete = None

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

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    mlflow.set_tracking_uri("http://0.0.0.0:5000")
    print(f'mlflow server: {mlflow.get_tracking_uri()}')
    args = get_args(argv)

    s2v = sent2vec.Sent2vecModel()
    s2v.load_model(args.sent2vec, inference_mode=True)


    mlflow.set_experiment('no_substituted_pairs_sent2vec')

    network = RelNet(out_dim=2)
    network.to(device)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    train_batches = load_batches(f'{args.dataset_dir}/train.vectors', s2v, args.batch_size)
    valid_batches = load_batches(f'{args.dataset_dir}/valid.vectors', s2v, args.batch_size)
    test_batches = load_batches(f'{args.dataset_dir}/test.vectors', s2v, args.batch_size)

    train_set_size = reduce((lambda x, y: x + len(y)), train_batches, 0)
    valid_set_size = reduce((lambda x, y: x + len(y)), valid_batches, 0)
    test_set_size = reduce((lambda x, y: x + len(y)), test_batches, 0)

    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('train_set_size', train_set_size)
    mlflow.log_param('valid_set_size', valid_set_size)
    mlflow.log_param('test_set_size', test_set_size)

    mlflow.log_param('epochs', args.epochs)

    best_valid_fscore = [0.0, 0.0]

    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch} / {args.epochs}')

        train_metrics = train(network, optimizer, loss_func, train_batches, device)
        print(f'Train:\n{train_metrics}')
        mlflow.log_metric(key='train_loss', value=train_metrics.loss, step=epoch)

        valid_metrics = evaluate(network, valid_batches, loss_func, device)
        print(f'Valid:\n{valid_metrics}')
        mlflow.log_metric(key='valid_loss', value=valid_metrics.loss, step=epoch)

        if valid_metrics.fscore[0] > best_valid_fscore[0] and valid_metrics.fscore[1] > best_valid_fscore[1]:
            best_valid_fscore = valid_metrics.fscore
            torch.save(network.state_dict(), args.model_name)
            # mlflow.pytorch.log_model(network, "torch_models")
            # mlflow.log_artifact(f'./{args.model_name}, 'path to minio')

    network = RelNet(out_dim=2)
    network.load(args.model_name)
    network.to(device)

    test_metrics = evaluate(network, test_batches, loss_func, device)
    print(f'\n\nTest: {test_metrics}')
    save_metrics(test_metrics, 'metrics.txt')

    mlflow.log_metric('loss', test_metrics.loss)
    mlflow.log_metric('accuracy', test_metrics.accuracy)
    mlflow.log_metric('precision_pos', test_metrics.precision[1])
    mlflow.log_metric('precision_neg', test_metrics.precision[0])
    mlflow.log_metric('recall_pos', test_metrics.recall[1])
    mlflow.log_metric('recall_neg', test_metrics.recall[0])
    mlflow.log_metric('fscore_pos', test_metrics.fscore[1])
    mlflow.log_metric('fscore_neg', test_metrics.fscore[0])


def train(network, optimizer, loss_func, batches, device):
    metrics = Metrics()

    network.train()
    for batch in batches:
        optimizer.zero_grad()

        labels, data = zip(*batch)
        target = Variable(torch.LongTensor(labels2idx(labels))).to(device)
        data = torch.FloatTensor([data])

        output = network(data.to(device)).squeeze(0)
        loss = loss_func(output, target)

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


if __name__ == "__main__":
    main()
