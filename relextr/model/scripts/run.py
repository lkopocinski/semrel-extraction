#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.autograd import Variable

from relextr.model import RelNet


def main():
    network = RelNet(out_dim=2)
    optimizer = Adagrad(network.parameters())
    loss_func = nn.CrossEntropyLoss()

    train_batches = load_batches('../../../data/train-52.vectors')
    valid_batches = load_batches('../../../data/valid-52.vectors')
    test_batches = load_batches('../../../data/test-52.vectors')

    best_valid_loss = float('inf')

    for epoch in range(10):
        train_loss, train_acc = train(network, optimizer, loss_func, train_batches)
        print('Train - Loss: {}, Accuracy: {}'.format(train_loss, train_acc))

        valid_loss, valid_acc = evaluate(network, valid_batches, loss_func)
        print('Valid - Loss: {}, Accuracy: {}'.format(valid_loss, valid_acc))
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(network.state_dict(), 'semrel.model.pt')

    test_acc = evaluate(network, test_batches, loss_func)
    print(' Test - Accuracy: {}'.format(test_acc))

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


def evaluate(network, batches, loss_function):
    eval_acc = 0.0
    eval_loss = 0.0
    network.eval()

    with torch.no_grad():
        for batch in batches:

            labels, data = zip(*batch)
            target = Variable(torch.LongTensor(labels2idx(labels)))
            data = torch.FloatTensor([data])

            output = network(data).squeeze(0)
            loss = loss_function(output, target)
            
            accuracy = compute_accuracy(output, target)
            eval_acc += accuracy
            eval_loss += loss.item()
    return eval_loss / len(batches), eval_acc / len(batches)

if __name__ == "__main__":
    main()
