import torch
from torch.autograd import Variable

from metrics import compute_accuracy_n, compute_precision_recall_fscore 


def remap_labels(labels, mapping):
    return [mapping[label] for label in labels if label in mapping]


def train_model(network, batches, loss_function, optimizer, labels2idx, device):
    epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f = 0.0, 0.0, 0.0, 0.0, 0.0    

    network.train()

    for batch in batches:
        optimizer.zero_grad()
        
        labels, data = zip(*batch)
        target = Variable(torch.FloatTensor(
            remap_labels(labels, labels2idx)
        )).to(device)
        data = torch.FloatTensor([data])

        output = network(data.to(device)).squeeze(0)
        loss = loss_function(output, target)

        loss.backward()
        optimizer.step()
        
        accuracy = compute_accuracy_n(output, target)
        precision, recall, fscore = compute_precision_recall_fscore(output, target)
        epoch_loss += loss.item()

        epoch_acc += accuracy
        epoch_prec += precision
        epoch_rec += recall
        epoch_f += fscore

        return {
            'loss': epoch_loss / len(batches),
            'accuracy': epoch_acc / len(batches),
            'precision': epoch_prec / len(batches),
            'recall': epoch_rec / len(batches),
            'fscore': epoch_f / len(batches)
        }


def evaluate_model(model, batches, loss_function, labels2idx, device):
    epoch_loss = 0
    epoch_accuracy = 0

    model.eval()

    with torch.no_grad():
        for batch in batches:
            output = model(batch.data.to(device)).squeeze(1)
            # _, predictions = torch.max(output, dim=1)

            targets = Variable(torch.FloatTensor(
                remap_labels(batch.labels, labels2idx))).to(device)

            loss = loss_function(output, targets)

            accuracy = compute_accuracy_n(output, targets)

            epoch_loss += loss.item()
            epoch_accuracy += accuracy
    return epoch_loss / len(batches), epoch_accuracy / len(batches)

