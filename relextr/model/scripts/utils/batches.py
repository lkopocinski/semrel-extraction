import random
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from torch.utils import data

random.seed(42)


class Dataset(data.Dataset):
    label2digit = {
        'no_relation': 0,
        'in_relation': 1,
    }

    @staticmethod
    def load_keys(path: Path):
        keys = {}
        with path.open('r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                keys[idx] = eval(line.strip())
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


class Sampler(data.Sampler):

    def __init__(self, dataset, set_type):
        self.dataset = dataset
        self.set_type = set_type
        # self.inverted_keys = {v: k for k, v in dataset.keys.items()}
        # self.domain_to_idx = self.indices_by_domain()

        self.train_indices = []
        self.valid_indices = []
        self.test_indices = []

    def __iter__(self):
        return iter(list(self.dataset.keys.keys()))
        # if self.set_type == 'train':
        #     return self.train_indices
        # elif self.set_type == 'valid':
        #     return self.valid_indices
        # elif self.set_type == 'test':
        #     return self.test_indices
        # else:
        #     raise KeyError(f'There is no data set for {self.set_type}')

    def __len__(self):
        return len(self.dataset.keys.keys())
        # if self.set_type == 'train':
        #     return len(self.train_indices)
        # elif self.set_type == 'valid':
        #     return len(self.valid_indices)
        # elif self.set_type == 'test':
        #     return len(self.test_indices)
        # else:
        #     raise KeyError(f'There is no data set for {self.set_type}')

    def indices_by_domain(self, domains=(112, 113, 115)):
        domain_to_idx = defaultdict(list)
        for k, i in self.inverted_keys.items():
            domain = k[0]
            if int(domain) in domains:
                domain_to_idx[domain].append(i)
        return domain_to_idx

    def make_data_sets(self, domains):
        data = defaultdict(list)

        for domain, indices in self.domain_to_idx:
            train, valid, test = self._split(indices)
            for key, index in self.inverted_keys:
                label = key[1]
                channel = key[3]

                if index in train:
                    data[(domain, 'train', label, channel)].append(index)
                elif index in valid:
                    data[(domain, 'valid', label, channel)].append(index)
                elif index in test:
                    data[(domain, 'test', label, channel)].append(index)

        for domain in domains:
            for settype in ['train', 'valid', 'test']:
                if settype == 'train':
                    indices_b = data[(domain, settype, 'in_relation', 'BRAND_NAME')]
                    indices_p = data[(domain, 'train', 'in_relation', 'PRODUCT_NAME')]
                    indices_n = data[(domain, 'train', 'no_relation', 'PRODUCT_NAME')]
                    # TODO: Losowanie negatywnych
                    self.train_indices.extend([indices_b, indices_p, indices_n])
                elif settype == 'valid':
                    indices_b = data[(domain, settype, 'in_relation', 'BRAND_NAME')]
                    indices_p = data[(domain, 'valid', 'in_relation', 'PRODUCT_NAME')]
                    indices_n = data[(domain, 'valid', 'no_relation', 'PRODUCT_NAME')]
                    # TODO: Losowanie negatywnych
                    self.valid_indices.extend([indices_b, indices_p, indices_n])
                if settype == 'test':
                    indices_b = data[(domain, settype, 'in_relation', 'BRAND_NAME')]
                    indices_p = data[(domain, 'test', 'in_relation', 'PRODUCT_NAME')]
                    indices_n = data[(domain, 'test', 'no_relation', 'PRODUCT_NAME')]
                    # TODO: Losowanie negatywnych
                    self.test_indices.extend([indices_b, indices_p, indices_n])

    def _split(self, indices):
        random.shuffle(indices)
        return self._chunk(indices)

    def _chunk(self, seq):
        avg = len(seq) / float(5)
        t_len = int(3 * avg)
        v_len = int(avg)
        return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]
