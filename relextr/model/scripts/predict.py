import torch
import numpy as np

from relextr.model import RelNet


network = RelNet()
network.load('../../../models/net-model.pt')

data = np.ones(3072)

mapping = {
        0: 'no-relation',
        1: 'in-relation'
}

output = network(torch.FloatTensor([data]))
_, predicted = torch.max(output, dim=1)
print(mapping[predicted.item()])
