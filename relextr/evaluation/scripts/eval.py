#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable

from relextr.model import RelNet


def main():
    network = RelNet(out_dim=2)
    network.load('./semrel.2d.static.fixed.balanced_2.model.pt')
    loss_func = nn.CrossEntropyLoss()
    test_batches = load_batches('/home/Projects/semrel-extraction/data/static_dataset_fixed/test.vectors')

    test_metrics = evaluate(network, test_batches, loss_func)
    print_metrics(test_metrics, 'Test')


def load_batches(datapath, batch_size=10):
    with open(datapath, encoding="utf-8") as ifile:
        dataset = []
        batch = []
        for ind, line in enumerate(ifile, 1):
            row = line.strip().split('\t')

            if len(row) < 3:
                continue

            cls = row[0]
            v1, v2 = np.array(eval(row[1])), np.array(eval(row[2]))
            if (ind % batch_size) == 0:
                dataset.append(batch)
                batch = []
            # vdiff = v1 - v2
            # batch.append((cls, np.concatenate([v1, v2, vdiff])))
            batch.append((cls, np.concatenate([v1, v2])))
        if batch:
            dataset.append(batch)
        return dataset


def labels2idx(labels):
    mapping = {
        'no_relation': 0,
        'in_relation': 1,
    }
    return [mapping[label] for label in labels if label in mapping]


def compute_accuracy(output, targets):
    _, predicted = torch.max(output, dim=1)
    return (predicted == targets).sum().item() / targets.shape[0]


def compute_precision_recall_fscore(output, targets):
    _, predicted = torch.max(output, dim=1)
    output = predicted.data.numpy()
    targets = targets.data.numpy()
    prec, rec, f, _ = precision_recall_fscore_support(targets, output, average=None, labels=[0, 1])
    return prec, rec, f


def print_metrics(metrics, prefix):
    print(f'\n\n{prefix}'
          f'\nLoss: {metrics["loss"]}, '
          f'\nAccuracy: {metrics["accuracy"]}, '
          f'\nPrecision: {metrics["precision"]}, '
          f'\nRecall: {metrics["recall"]}, '
          f'\nFscore: {metrics["fscore"]}')


def evaluate(network, batches, loss_function):
    eval_loss = 0.0
    eval_acc = 0.0
    eval_prec = (0.0, 0.0)
    eval_rec = (0.0, 0.0)
    eval_f = (0.0, 0.0)
    network.eval()

    with torch.no_grad():
        for batch in batches:
            labels, data = zip(*batch)
            target = Variable(torch.LongTensor(labels2idx(labels)))
            data = torch.FloatTensor([data])

            output = network(data).squeeze(0)
            loss = loss_function(output, target)

            accuracy = compute_accuracy(output, target)
            precision, recall, fscore = compute_precision_recall_fscore(output, target)
            eval_loss += loss.item()

            eval_acc += accuracy
            eval_prec = tuple(sum(x) for x in zip(eval_prec, precision))
            eval_rec = tuple(sum(x) for x in zip(eval_rec, recall))
            eval_f = tuple(sum(x) for x in zip(eval_f, fscore))

    return {
        'loss': eval_loss / len(batches),
        'accuracy': eval_acc / len(batches),
        'precision': [prec / len(batches) for prec in eval_prec],
        'recall': [rec / len(batches) for rec in eval_rec],
        'fscore': [f / len(batches) for f in eval_f]
    }


if __name__ == "__main__":
    main()
