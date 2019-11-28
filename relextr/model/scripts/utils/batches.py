import random
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from torch.utils import data

random.seed(42)


class Dataset(data.dataset):
    label2digit = {
        'no_relation': 0,
        'in_relation': 1,
    }

    @staticmethod
    def load_keys(path: Path):
        keys = {}
        with path.open('r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                row = line.strip().split('\t')
                keys[idx] = tuple(row)
        return keys

    def __init__(self, vectors_models: List[str], keys: dict):
        self.vectors_models = vectors_models
        self.keys = keys

        self.vectors = [torch.load(model) for model in self.vectors_models]
        self.vectors = torch.cat(self.vectors, dim=1)

    @property
    def vector_size(self):
        return self.vectors.shape[-1]

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, index: int):
        X = self.vectors[index]
        y = self.label2digit[self.keys[index][1]]
        return X, y


class Sampler(object):

    def __init__(self, dataset, set_type):
        self.dataset = dataset
        self.set_type = set_type
        self.invert_keys = {v: k for k, v in dataset.keys.items()}
        self.domain_to_idx = self.indices_by_domain()
        self.domain_to_settype = self.indices_by_domain_by_settype()

        self.train_indices = []
        self.valid_indices = []
        self.test_indices = []

    def __iter__(self):
        if self.set_type == 'train':
            return self.train_indices
        elif self.set_type == 'valid':
            return self.valid_indices
        elif self.set_type == 'test':
            return self.test_indices
        else:
            raise KeyError(f'There is no data set for {self.set_type}')

    def __len__(self):
        if self.set_type == 'train':
            return len(self.train_indices)
        elif self.set_type == 'valid':
            return len(self.valid_indices)
        elif self.set_type == 'test':
            return len(self.test_indices)
        else:
            raise KeyError(f'There is no data set for {self.set_type}')

    def indices_by_domain(self, domains=(112, 113, 115)):
        domain_to_idx = defaultdict(list)
        for k, i in self.invert_keys.items():
            domain = k[0]
            if int(domain) in domains:
                domain_to_idx[domain] = i
        return domain_to_idx

    def indices_by_domain_by_settype(self):
        domain_to_settype = defaultdict(list)
        for k, v in self.domain_to_idx.items():
            domain_to_settype[k] = self._split(v)
        return domain_to_settype

    def indices_by_label(self):
        # TODO: negative, positive
        pass

    def _split(self, indices):
        random.shuffle(indices)
        return self._chunk(indices)

    def _chunk(self, seq):
        avg = len(seq) / float(5)
        t_len = int(3 * avg)
        v_len = int(avg)
        return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]
