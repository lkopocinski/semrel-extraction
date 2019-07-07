#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.autograd import Variable

from base.model import RelNet


def main():
    network = RelNet(out_dim=2)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    train_batches = load_batches('./data/train-52.vectors')
    valid_batches = load_batches('./data/valid-52.vectors')
    test_batches = load_batches('./data/test-52.vectors')

    for epoch in range(100):
        train_loss, train_acc = train(network, optimizer, loss_func, train_batches)
        print('Train - Loss: {}, Accuracy: {}'.format(train_loss, train_acc))

        valid_acc = evaluate(network, valid_batches)
        print('Valid - Accuracy: {}'.format(valid_acc))

    test_acc = evaluate(network, test_batches)
    print(' Test - Accuracy: {}'.format(test_acc))

    # extract the layer with embedding
    embeddings = network.extract_layer_weights('f2')
    # save the embeddings


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
    with open(datapath) as ifile:
        dataset = []
        batch = []
        for ind, line in enumerate(ifile, 1):
            row = line.strip().split('\t')
            cls = row[0]
            v1, v2 = np.array(eval(row[1])), np.array(eval(row[2]))
            if (ind % batch_size) == 0:
                dataset.append(batch)
                batch = []
            vdiff = v1 - v2
            batch.append((cls, np.concatenate([v1, v2, vdiff])))
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


def train(network, optimizer, loss_func, batches):
    # TODO: Cosine Embedding Loss, ontologia jako regularyzator!
    ep_loss, ep_acc = 0.0, 0.0
    network.train()

    for batch in batches:
        optimizer.zero_grad()

        labels, data = zip(*batch)
        target = Variable(torch.LongTensor(labels2idx(labels)))
        data = torch.FloatTensor([data])

        output = network(data).squeeze(0)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(output, target)
        ep_loss += loss.item()
        ep_acc += accuracy
    return ep_loss / len(batches), ep_acc / len(batches)


def evaluate(network, batches):
    eval_acc = 0.0
    network.eval()

    with torch.no_grad():
        for batch in batches:

            labels, data = zip(*batch)
            target = Variable(torch.LongTensor(labels2idx(labels)))
            data = torch.FloatTensor([data])

            output = network(data).squeeze(0)
            
            accuracy = compute_accuracy(output, target)
            eval_acc += accuracy
    return eval_acc / len(batches)


if __name__ == "__main__":
    main()
