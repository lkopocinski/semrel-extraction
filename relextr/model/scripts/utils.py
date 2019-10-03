import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


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
    prec, rec, f, _ = precision_recall_fscore_support(targets, output, average='weighted', labels=[0, 1])
    return prec, rec, f


class Metrics:

    def __init__(self):
        self._loss = 0.0
        self._acc = 0.0
        self._prec = (0.0, 0.0)
        self._rec = (0.0, 0.0)
        self._f = (0.0, 0.0)
        self.batches = 0

    def update(self, loss, accuracy, precision, recall, fscore, batches):
        self._loss += loss
        self._acc += accuracy
        self._prec = tuple(sum(x) for x in zip(self._prec, precision))
        self._rec = tuple(sum(x) for x in zip(self._rec, recall))
        self._f = tuple(sum(x) for x in zip(self._f, fscore))
        self.batches = batches

    @property
    def loss(self):
        return self._loss / self.batches

    @property
    def accuracy(self):
        return self._acc / self.batches

    @property
    def precision(self):
        return [prec / self.batches for prec in self._prec]

    @property
    def recall(self):
        return [rec / self.batches for rec in self._rec]

    @property
    def fscore(self):
        return [f / self.batches for f in self._f]

    def __str__(self):
        return f'Loss: {self._loss} ' \
            f'Accuracy: {self.accuracy} ' \
            f'Precision: {self.precision} ' \
            f'Recall: {self.recall} ' \
            f'Fscore: {self.fscore}'


def save_metrics(metrics, path):
    with open(path, 'w', encoding='utf-8') as out_file:
        out_file.write(f'{metrics}')
