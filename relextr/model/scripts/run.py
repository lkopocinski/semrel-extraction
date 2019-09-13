#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.autograd import Variable

from relextr.model import RelNet

EPOCHS_QUANTITY = 30

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)


def main():
    network = RelNet(out_dim=2)
    network.to(device)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    # train_batches = load_batches('/home/Projects/semrel-extraction/data/static_dataset_fixed_arek/train.vectors')
    # valid_batches = load_batches('/home/Projects/semrel-extraction/data/static_dataset_fixed_arek/valid.vectors')
    # test_batches = load_batches('/home/Projects/semrel-extraction/data/static_dataset_fixed_arek/test.vectors')
    train_batches = load_batches('/home/Projects/semrel-extraction/data/static_dataset_fixed_arek/train1k.vectors')
    valid_batches = load_batches('/home/Projects/semrel-extraction/data/static_dataset_fixed_arek/valid100.vectors')
    test_batches = load_batches('/home/Projects/semrel-extraction/data/static_dataset_fixed_arek/test100.vectors')

    best_valid_loss = float('inf')

    for epoch in range(EPOCHS_QUANTITY):
        print(f'\nEpoch: {epoch} / {EPOCHS_QUANTITY}')

        train_metrics = train(network, optimizer, loss_func, train_batches, device)
        print_metrics(train_metrics, 'Train')

        valid_metrics = evaluate(network, valid_batches, loss_func, device)
        print_metrics(valid_metrics, 'Valid')

        valid_loss = valid_metrics['loss']
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(network.state_dict(), 'semrel.2d.static.fixed.arek.model.pt')

    test_metrics = evaluate(network, test_batches, loss_func, device)
    print_metrics(test_metrics, '\n\nTest')

    # extract the layer with embedding
    # embeddings = network.extract_layer_weights('f2')
    # todo: save the embeddings


def load_batches_example(datafile):
    # for now - mock it
    dataset = []
    batch = []
    for i in range(100):
        if i % 6 == 0 and i != 0:
            random.shuffle(batch)
            dataset.append(batch)
            batch = []
        vec = np.random.normal(2.0, 1.0, 600)
        batch.append(('r1', vec))

        vec = np.random.normal(-1.0, 1.0, 600)
        batch.append(('r2', vec))

        vec = np.random.normal(10, 1.0, 600)
        batch.append(('r3', vec))

    return dataset


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
    prec, rec, f, _ = precision_recall_fscore_support(targets, output, average='weighted', labels=[0, 1])
    return prec, rec, f


def print_metrics(metrics, prefix):
    print(f'{prefix} - Loss: {metrics["loss"]}, '
          f'Accuracy: {metrics["accuracy"]}, '
          f'Precision: {metrics["precision"]}, '
          f'Recall: {metrics["recall"]}, '
          f'Fscore: {metrics["fscore"]}')


def train(network, optimizer, loss_func, batches, device):
    # TODO: Cosine Embedding Loss, ontologia jako regularyzator!
    ep_loss, ep_acc, ep_prec, ep_rec, ep_f = 0.0, 0.0, 0.0, 0.0, 0.0
    network.train()

    for batch in batches:
        optimizer.zero_grad()

        labels, data = zip(*batch)
        target = Variable(torch.LongTensor(labels2idx(labels)))
        data = torch.FloatTensor([data])
        data.to(device)

        output = network(data).squeeze(0)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(output, target)
        precision, recall, fscore = compute_precision_recall_fscore(output, target)
        ep_loss += loss.item()

        ep_acc += accuracy
        ep_prec += precision
        ep_rec += recall
        ep_f += fscore

    return {
        'loss': ep_loss / len(batches),
        'accuracy': ep_acc / len(batches),
        'precision': ep_prec / len(batches),
        'recall': ep_rec / len(batches),
        'fscore': ep_f / len(batches)
    }


def evaluate(network, batches, loss_function, device):
    eval_loss, eval_acc, eval_prec, eval_rec, eval_f = 0.0, 0.0, 0.0, 0.0, 0.0
    network.eval()

    with torch.no_grad():
        for batch in batches:
            labels, data = zip(*batch)
            target = Variable(torch.LongTensor(labels2idx(labels)))
            data = torch.FloatTensor([data])
            data.to(device)

            output = network(data).squeeze(0)
            loss = loss_function(output, target)

            accuracy = compute_accuracy(output, target)
            precision, recall, fscore = compute_precision_recall_fscore(output, target)
            eval_loss += loss.item()

            eval_acc += accuracy
            eval_prec += precision
            eval_rec += recall
            eval_f += fscore

    return {
        'loss': eval_loss / len(batches),
        'accuracy': eval_acc / len(batches),
        'precision': eval_prec / len(batches),
        'recall': eval_rec / len(batches),
        'fscore': eval_f / len(batches)
    }


if __name__ == "__main__":
    main()
