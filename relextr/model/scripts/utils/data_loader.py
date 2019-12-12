import random
from collections import defaultdict
from typing import List

import torch
from torch.utils import data


class BrandProductDataset(data.Dataset):
    label2digit = {
        'no_relation': 0,
        'in_relation': 1,
    }

    def __init__(self, keys_file: str, vectors_files: List[str]):
        self.keys = self._load_keys(keys_file)
        self.vectors = [torch.load(file) for file in vectors_files]
        self.vectors = torch.cat(self.vectors, dim=1)

    @staticmethod
    def _load_keys(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            return {idx: tuple(line.strip().split('\t')) for idx, line in enumerate(f)}

    @property
    def vector_size(self):
        return self.vectors.shape[-1]

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, index: int):
        x = self.vectors[index]
        y = self.label2digit[self.keys[index][1]]
        return x, y


class DatasetGenerator:

    def __init__(self, dataset, random_seed=42):
        self.ds = dataset
        random.seed(random_seed)

    def _filter_indices_by_channels(self, indices, channels):
        return {idx for idx in indices if (self.ds.keys[idx][5] in channels or
                                           self.ds.keys[idx][6] in channels)}

    def _split(self, indices):
        random.shuffle(indices)
        return self._chunk(indices)

    def _chunk(self, seq):
        avg = len(seq) / float(5)
        t_len = int(3 * avg)
        v_len = int(avg)
        return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]

    def _generate(self, indices: List, balanced: bool, lexical_split: bool):
        """ The data is splitted to train, dev, and test. """
        if not balanced:
            return self._split(indices)
        # ok, lets try to balance the data (positives vs negatives)
        # 2 cases to cover: i) B-N, P-N, and ii) N-N
        positives = {idx for idx in indices if self.ds.keys[idx][1] == 'in_relation'}
        negatives = {idx for idx in indices if self.ds.keys[idx][1] == 'no_relation'}

        # take the negatives connected with Bs or Ps
        negatives_bps = self._filter_indices_by_channels(negatives, ('BRAND_NAME', 'PRODUCT_NAME'))
        negatives_nns = negatives - negatives_bps

        # balance the data (take 2 times #positives of negative examples)
        if negatives_bps and len(negatives_bps) >= len(positives):
            negatives_bps = random.sample(sorted(negatives_bps), len(positives))
        if negatives_nns and len(negatives_nns) >= len(positives):
            negatives_nns = random.sample(sorted(negatives_nns), len(positives))

        negatives = set(negatives_bps + negatives_nns)
        if not lexical_split:
            return self._split(list(positives | negatives))

        # ok, lexical split... Lets take all the brands and split the dataset
        return self._split_lexically(positives, negatives)

    def _split_lexically(self, positives: set, negatives: set):
        # 9 - the position of left argument,
        # 10 - the position of right argument
        # 5 - channel name for left argument
        # 6 - channel name for right argument
        train, valid, test = [], [], []
        nns_and_nps_indices = []
        brand_indices = defaultdict(list)
        for idx in sorted(positives | negatives):
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

    def generate_datasets(self, balanced: bool, lexical_split: bool, in_domain: str, out_domain: str = None):
        if in_domain:
            indices = [idx for idx, desc in self.ds.keys.items() if desc[0] == in_domain]
        elif out_domain:
            raise NotImplementedError(f'Out domain dataset split not implemented.')
        else:
            indices = self.ds.keys.keys()

        return self._generate(indices, balanced, lexical_split)


def get_loaders(data_dir: str,
                keys_file: str,
                vectors_files: List[str],
                batch_size: int,
                balanced: bool = False,
                lexical_split: bool = False,
                in_domain: str = None,
                out_domain: str = None,
                random_seed: int = 42,
                shuffle: bool = True,
                num_workers: int = 8,
                pin_memory: bool = False):
    dataset = BrandProductDataset(
        keys_file=f'{data_dir}/{keys_file}',
        vectors_files=[f'{data_dir}/{file}' for file in vectors_files],
    )

    ds_generator = DatasetGenerator(dataset, random_seed)
    train_indices, valid_indices, test_indices = ds_generator.generate_datasets(balanced, lexical_split, in_domain)

    train_loader = data.DataLoader(
        dataset, batch_size, shuffle, train_indices, num_workers, pin_memory=pin_memory,
    )
    valid_loader = data.DataLoader(
        dataset, batch_size, shuffle, valid_indices, num_workers, pin_memory=pin_memory,
    )
    test_loader = data.DataLoader(
        dataset, batch_size, shuffle, test_indices, num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader, dataset.vector_size
