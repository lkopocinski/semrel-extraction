import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Metrics:

    def __init__(self):
        self._loss = 0.0
        self.batches = 0

        self._predicted = np.array([])
        self._targets = np.array([])

    def update(self, predicted, targets, loss, batches):
        _, predicted = torch.max(predicted, dim=1)
        predicted = predicted.data.numpy()
        targets = targets.data.numpy()

        self._predicted = np.concatenate((self._predicted, predicted))
        self._targets = np.concatenate((self._targets, targets))

        self._loss += loss
        self.batches = batches

    @property
    def loss(self):
        return self._loss / self.batches

    @property
    def accuracy(self):
        return accuracy_score(self._targets, self._predicted)

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
        return f'\tLoss: {self._loss}' \
            f'\n\tAccuracy: {self.accuracy}' \
            f'\n\tPrecision: {self.precision}' \
            f'\n\tRecall: {self.recall}' \
            f'\n\tFscore: {self.fscore}'


def save_metrics(metrics, path):
    with open(path, 'w', encoding='utf-8') as out_file:
        out_file.write(f'{metrics}')
