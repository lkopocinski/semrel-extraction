import random
from collections import defaultdict
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
    def load_keys(path):
        keys = {}
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                keys[idx] = eval(line.strip())
        return keys

    def __init__(self, vectors_models: List[str], keys: dict):
        self.vectors_models = vectors_models
        self.keys = keys

        self.vectors = [torch.load(model) for model in self.vectors_models]
        self.vectors = torch.cat(self.vectors, dim=1)
        # self.vectors = [i for i in range(len(self.keys))]

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

    def __init__(self, dataset, balanced=False, lexical_split=True, in_domain=None, out_domain=None):
        self.ds = dataset
        self._set_type = None

        self.train_indices = []
        self.valid_indices = []
        self.test_indices = []

        self.generate_dataset(balanced, lexical_split, in_domain, out_domain)

    @property
    def set_type(self):
        return self._set_type

    @set_type.setter
    def set_type(self, val):
        self._set_type = val if val in ('train', 'valid', 'test') else None

    def __iter__(self):
        if self._set_type == 'train':
            return iter(self.train_indices)
        elif self._set_type == 'valid':
            return iter(self.valid_indices)
        elif self._set_type == 'test':
            return iter(self.test_indices)
        else:
            raise KeyError(f'There is no data set for {self._set_type}')

    def __len__(self):
        if self._set_type == 'train':
            return len(self.train_indices)
        elif self._set_type == 'valid':
            return len(self.valid_indices)
        elif self._set_type == 'test':
            return len(self.test_indices)
        else:
            raise KeyError(f'There is no data set for {self._set_type}')

    def _filter_indices_by_channels(self, indices, channels):
        return [idx for idx in indices if (self.ds.keys[idx][5] in channels or
                                           self.ds.keys[idx][6] in channels)]

    def _ds_mixed(self, balanced=False, lexical_split=True, in_domain=None, out_domain=None):
        """ Just ignore the structure of the data: we want a mixed dataset with
        all domains together. The data is splitted to train, dev, and test. """
        if in_domain:
            indices = [idx for idx, desc in self.ds.keys.items()
                       if desc[0] == in_domain]
        elif out_domain:
            return self._ds_domain_out(out_domain)
        else:
            indices = self.ds.keys.keys()

        if not balanced:
            return self._split(indices)
        # ok, lets try to balance the data (positives vs negatives)
        # 2 cases to cover: i) B-N, P-N, and ii) N-N
        positives = {idx for idx in indices if self.ds.keys[idx][1] == 'in_relation'}
        negatives = {idx for idx in indices if self.ds.keys[idx][1] == 'no_relation'}

        # take the negatives connected with Bs or Ps
        negatives_bps = set(self._filter_indices_by_channels(
            negatives, ('BRAND_NAME', 'PRODUCT_NAME')))
        negatives_nns = negatives.difference(negatives_bps)

        # balance the data (take 2 times #positives of negative examples)
        if negatives_bps and len(negatives_bps) >= len(positives):
            negatives_bps = random.sample(negatives_bps, len(positives))
        if negatives_nns and len(negatives_nns) >= len(positives):
            negatives_nns = random.sample(negatives_nns, len(positives))

        negatives = set.union(set(negatives_bps), set(negatives_nns))
        if not lexical_split:
            return self._split(list(positives.union(negatives)))

        # ok, lexical split... Lets take all the brands and split the dataset
        return self._lexical_split(positives, negatives)

    def _lexical_split(self, positives, negatives):
        # 9 - the position of left argument,
        # 10 - the position of right argument
        # 5 - channel name for left argument
        # 6 - channel name for right argument
        train, valid, test = [], [], []
        nns_and_nps_indices = list()

        brand_indices = defaultdict(list)
        for idx in positives.union(negatives):
            brand = None
            if self.ds.keys[idx][5] == 'BRAND_NAME':
                brand = self.ds.keys[idx][9]
            elif self.ds.keys[idx][6] == 'BRAND_NAME':
                brand = self.ds.keys[idx][10]
            else:
                nns_and_nps_indices.append(idx)
            if brand:
                brand_indices[brand].append(idx)

        n_brand_indices = sum([len(brand_indices[b]) for b in brand_indices])

        # split equally starting from the least frequent brands
        counter = 0
        for brand in sorted(brand_indices, key=lambda k: len(brand_indices[k])):
            # if some brand has more than 50% of examples -> add it to train
            if len(brand_indices[brand]) > (0.5 * n_brand_indices):
                counter = 0
            if counter % 3 == 0:
                train.extend(brand_indices[brand])
            elif counter % 3 == 1:
                valid.extend(brand_indices[brand])
            elif counter % 3 == 2:
                test.extend(brand_indices[brand])
            counter += 1

        # use held_out indices of type N-N and N-P and split them
        # to make our datasets more like 3:1:1
        tr, vd, ts = self._split(nns_and_nps_indices)
        train.extend(tr)
        valid.extend(vd)
        test.extend(ts)
        return train, valid, test

    def _ds_domain_out(self, domain):
        """ Lets remind ourselves that the stucture of our `keys` with  indices
        and metadata contains:

            domain, label, doc_id, first_sent_id, second_sent_id, ...

        We generate splitted data (train, dev, test) by examining the domain of
        our examples, omiting the examples annotated with given `out_domain`. First
        take all of in_domain examples and generate subsets for train, and dev.
        Then the test set is built from all `out_domain` examples.
        """
        in_domain = [idx for idx, desc in self.ds.keys.items()
                     if desc[0] != domain]
        out_domain = [idx for idx, desc in self.ds.keys.items()
                      if desc[0] == domain]

        # ok, lets try to balance the data (positives vs negatives)
        # 2 cases to cover: i) B-N, P-N, and ii) N-N
        positives = {idx for idx in in_domain if self.ds.keys[idx][1] == 'in_relation'}
        negatives = {idx for idx in in_domain if self.ds.keys[idx][1] == 'no_relation'}

        # take the negatives connected with Bs or Ps
        negatives_bps = set(self._filter_indices_by_channels(
            negatives, ('BRAND_NAME', 'PRODUCT_NAME')))
        negatives_nns = negatives.difference(negatives_bps)

        # balance the data (take 2 times #positives of negative examples)
        if negatives_bps and len(negatives_bps) >= len(positives):
            negatives_bps = random.sample(negatives_bps, len(positives))
        if negatives_nns and len(negatives_nns) >= len(positives):
            negatives_nns = random.sample(negatives_nns, len(positives))

        negatives = set.union(set(negatives_bps), set(negatives_nns))
        in_domain_train, in_domain_valid, in_domain_test = self._split(list(positives.union(negatives)))

        train = in_domain_train
        valid = in_domain_valid + in_domain_test
        test = out_domain

        return train, valid, test

    def generate_dataset(self, balanced=False, lexical_split=True, in_domain=None, out_domain=None):
        self.train_indices, self.valid_indices, self.test_indices = self._ds_mixed(balanced, lexical_split, in_domain,
                                                                                   out_domain)

    def _split(self, indices):
        random.shuffle(indices)
        return self._chunk(indices)

    def _chunk(self, seq):
        avg = len(seq) / float(5)
        t_len = int(3 * avg)
        v_len = int(avg)
        return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]
