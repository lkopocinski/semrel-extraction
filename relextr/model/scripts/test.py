#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn
from relnet import RelNet
from torch.autograd import Variable
from utils import load_batches, labels2idx, Metrics, save_metrics
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
    parser.add_argument('-n', '--model_name', required=True, type=str, help="Load pre-trained model.")
    parser.add_argument('-b', '--batch_size', required=True, type=int,
                        help="Batch size.")
    parser.add_argument('-d', '--dataset_dir', required=True, type=str,
                        help="Directory with test dataset.")
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
    # mlflow.start_run(run_id=None, experiment_id=None, run_name='lexical_filter', nested=False)
    mlflow.set_tag(key='evaluation', value='lexical_filter')

    loss_func = nn.CrossEntropyLoss()

    test_batches = load_batches(f'{args.dataset_dir}/test.vectors_', s2v, args.batch_size)
    test_set_size = reduce((lambda x, y: x + len(y)), test_batches, 0)

    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('test_set_size', test_set_size)

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

    # mlflow.end_run()


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
