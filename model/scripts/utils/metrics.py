from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Metrics:

    def __init__(self):
        self._loss = 0.0
        self._batches = 0

        self._predicted = np.array([])
        self._targets = np.array([])

    def update(self, predicted, targets, loss, batches: int):
        _, predicted = torch.max(predicted, dim=1)
        predicted = predicted.data.numpy()
        targets = targets.data.numpy()

        self._predicted = np.concatenate((self._predicted, predicted))
        self._targets = np.concatenate((self._targets, targets))

        self._loss += loss
        self._batches = batches

    @property
    def loss(self) -> float:
        return self._loss / self._batches

    @property
    def accuracy(self) -> float:
        return accuracy_score(self._targets, self._predicted)

    @property
    def precision(self) -> List[float]:
        return precision_score(self._targets, self._predicted, average=None)

    @property
    def recall(self) -> List[float]:
        return recall_score(self._targets, self._predicted, average=None)

    @property
    def fscore(self) -> List[float]:
        return f1_score(self._targets, self._predicted, average=None)

    def __str__(self):
        return f'\tLoss: {self._loss}' \
               f'\n\tAccuracy: {self.accuracy}' \
               f'\n\tPrecision: {self.precision}' \
               f'\n\tRecall: {self.recall}' \
               f'\n\tFscore: {self.fscore}'


class NerMetrics:

    def __init__(self):
        self._predicted = []
        self._targets = []
        self._ner_from = []
        self._ner_to = []
        self._ner_predicted = []

    def append(self, predicted, targets, ner_from, ner_to):
        _, predicted = torch.max(predicted, dim=1)
        predicted = predicted.data.numpy()
        targets = targets.data.numpy()

        self._predicted = np.append(self._predicted, predicted)
        self._targets = np.append(self._targets, targets)
        self._ner_from.extend(ner_from)
        self._ner_to.extend(ner_to)
        self.predict_ner(predicted, targets, ner_from, ner_to)

    def predict_ner(self, predicted, targets, ner_from, ner_to):
        for ner_from, ner_to, target, prediction in zip(ner_from, ner_to, targets, predicted):
            neither_ner = not (eval(ner_from) or eval(ner_to))
            if neither_ner and target == 0 and prediction == 1:
                self._ner_predicted.append(0)
            else:
                self._ner_predicted.append(prediction)

    @property
    def accuracy(self) -> float:
        return accuracy_score(self._targets, self._ner_predicted)

    @property
    def precision(self) -> List[float]:
        return precision_score(self._targets, self._ner_predicted, average=None)

    @property
    def recall(self) -> List[float]:
        return recall_score(self._targets, self._ner_predicted, average=None)

    @property
    def fscore(self) -> List[float]:
        return f1_score(self._targets, self._ner_predicted, average=None)

    def __str__(self):
        return f'\tAccuracy: {self.accuracy}' \
               f'\n\tPrecision: {self.precision}' \
               f'\n\tRecall: {self.recall}' \
               f'\n\tFscore: {self.fscore}'
