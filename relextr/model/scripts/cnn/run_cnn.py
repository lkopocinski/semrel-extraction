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

from model import CNN

def main():
    train_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5/train.vectors')
    valid_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5/valid.vectors')
    test_batches = load_batches('/home/Projects/semrel-extraction/data/data_model_5/test.vectors')

    mapping = {
        'no_relation': 0,
        'in_relation': 1,
    }

    params = {'emb_dim': 3072, 'n_filters': 128, 'filter_sizes': (4, 8, 16, 32),
            'out_dim': 2, 'dropout': 0.5
    }
    
    model = CNN(**params)
    loss_function = nn.BCEWithLogitsLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    loss_function.to(device)
  
    optimizer = Adagrad(model.parameters())
 
    # training 
    best_valid_loss = float('inf')
    for epoch in range(20):
        train_metrics = train(network, optimizer, loss_func, train_batches)
        print_metrics(train_metrics, 'Train')

        valid_metrics = evaluate(network, valid_batches, loss_func)
        print_metrics(valid_metrics, 'Valid')

        valid_loss = valid_metrics['loss']
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(network.state_dict(), 'semrel.model_5.pt')

    test_metrics = evaluate(network, test_batches, loss_func)
    print_metrics(test_metrics, 'Test')

    # extract the layer with embedding
    # embeddings = network.extract_layer_weights('f2')
    # todo: save the embeddings

def load_batches(datapath, batch_size=15):
    with open(datapath, encoding="utf-8") as ifile:
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


def compute_precision_recall_fscore(output, targets):
    _, predicted = torch.max(output, dim=1)
    output = predicted.data.numpy()
    targets = targets.data.numpy()
    prec, rec, f, _ = precision_recall_fscore_support(targets, output, average='weighted')
    return prec, rec, f


def print_metrics(metrics, prefix):
    print(f'{prefix} - Loss: {metrics["loss"]}, '
          f'Accuracy: {metrics["accuracy"]}, '
          f'Precision: {metrics["precision"]}, '
          f'Recall: {metrics["recall"]}, '
          f'Fscore: {metrics["fscore"]}')


def train(network, optimizer, loss_func, batches):
    # TODO: Cosine Embedding Loss, ontologia jako regularyzator!
    ep_loss, ep_acc, ep_prec, ep_rec, ep_f = 0.0, 0.0, 0.0, 0.0, 0.0
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


def evaluate(network, batches, loss_function):
    eval_loss, eval_acc, eval_prec, eval_rec, eval_f = 0.0, 0.0, 0.0, 0.0, 0.0
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
