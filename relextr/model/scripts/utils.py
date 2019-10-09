import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score


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


class Metrics:

    def __init__(self):
        self._loss = 0.0
        self._acc = 0.0
        self._prec = (0.0, 0.0)
        self._rec = (0.0, 0.0)
        self._f = (0.0, 0.0)
        self.batches = 0

        self._predicted = []
        self._targets = []

    def update(self, loss, accuracy, precision, recall, fscore, batches):
        self._loss += loss
        self._acc += accuracy
        self._prec = tuple(sum(x) for x in zip(self._prec, precision))
        self._rec = tuple(sum(x) for x in zip(self._rec, recall))
        self._f = tuple(sum(x) for x in zip(self._f, fscore))
        self.batches = batches

    def update_count(self, predicted, targets):
        _, predicted = torch.max(predicted, dim=1)
        predicted = predicted.data.numpy()
        targets = targets.data.numpy()

        self._predicted.extend(predicted)
        self._targets.extend(targets)

    @property
    def loss(self):
        return self._loss / self.batches

    @property
    def accuracy(self):
        return self._acc / self.batches

    @property
    def precision(self):
        return precision_score(self._targets, self._predicted, average=None)

    @property
    def recall(self):
        return recall_score(self._targets, self._predicted, average=None)

    @property
    def fscore(self):
        return f1_score(self._targets, self._predicted, average=None)

    def __str__(self):
        return f'\tLoss: {self._loss} ' \
            f'\n\tAccuracy: {self.accuracy} ' \
            f'\n\tPrecision: {self.precision} ' \
            f'\n\tRecall: {self.recall} ' \
            f'\n\tFscore: {self.fscore}'


def save_metrics(metrics, path):
    with open(path, 'w', encoding='utf-8') as out_file:
        out_file.write(f'{metrics}')
