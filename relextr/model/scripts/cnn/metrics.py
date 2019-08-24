import torch
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

def compute_accuracy_n(predicted, targets):
    pred = predicted.argmax(dim=1)
    true = targets.argmax(dim=1)
    return (pred == true).sum().item() / true.shape[0]


def compute_precision_recall_fscore(output, targets):
    _, predicted = torch.max(output, dim=1)
    output = predicted.data.numpy()
    targets = targets.data.numpy()
    prec, rec, f, _ = precision_recall_fscore_support(targets, output, average='weighted')
    return prec, rec, f


