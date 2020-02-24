import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple

import torch
from torch.utils import data

CHANNELS = ('BRAND_NAME', 'PRODUCT_NAME')


class BrandProductDataset(data.Dataset):
    label2digit = {
        'no_relation': 0,
        'in_relation': 1,
    }

    def __init__(self, keys_file: str, vectors_files: List[str]):
        self.keys = self._load_keys(Path(keys_file))
        self.vectors = [torch.load(file) for file in vectors_files]
        self.vectors = torch.cat(self.vectors, dim=1)

    @staticmethod
    def _load_keys(path: Path) -> Dict[int, str]:
        with path.open('r', encoding='utf-8') as file:
            return {index: tuple(line.strip().split('\t'))
                    for index, line in enumerate(file)}

    @property
    def vector_size(self) -> int:
        return self.vectors.shape[-1]

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, index: int):
        label = self.keys[index][0]

        x = self.vectors[index]
        y = self.label2digit[label]
        return x, y


class DatasetGenerator:

    def __init__(self, dataset: data.Dataset, random_seed: int = 42):
        self.dataset = dataset
        random.seed(random_seed)

    def _filter_indices_by_channels(self, indices: Set[int], channels) -> Set:
        return {index for index in indices if (self.dataset.keys[index][4] in channels or
                                               self.dataset.keys[index][8] in channels)}

    def _split(self, indices) -> Tuple[List, List, List]:
        random.shuffle(indices)
        return self._chunk(indices)

    def _chunk(self, sequence) -> Tuple[List, List, List]:
        avg = len(sequence) / float(5)
        t_len = int(3 * avg)
        v_len = int(avg)
        return sequence[0:t_len], sequence[t_len:t_len + v_len], sequence[t_len + v_len:]

    def _generate(self, indices: List, balanced: bool, lexical_split: bool) -> Tuple[List, List, List]:
        """ The data is split to train, dev, and test. """
        if not balanced:
            return self._split(indices)
        # ok, lets try to balance the data (positives vs negatives)
        # 2 cases to cover: i) B-N, P-N, and ii) N-N
        positives = {index for index in indices if self.dataset.keys[index][0] == 'in_relation'}
        negatives = {index for index in indices if self.dataset.keys[index][0] == 'no_relation'}

        # take the negatives connected with Bs or Ps
        negatives_bps = self._filter_indices_by_channels(negatives, CHANNELS)
        negatives_nns = negatives - negatives_bps

        # balance the data (take 2 times #positives of negative examples)
        if negatives_bps and len(negatives_bps) >= len(positives):
            negatives_bps = random.sample(sorted(negatives_bps), len(positives))
        if negatives_nns and len(negatives_nns) >= len(positives):
            negatives_nns = random.sample(sorted(negatives_nns), len(positives))

        negatives = set(negatives_bps).union(set(negatives_nns))
        if not lexical_split:
            return self._split(list(positives | negatives))

        # ok, lexical split... Lets take all the brands and split the dataset
        return self._split_lexically(positives, negatives)

    def _split_lexically(self, positives: Set, negatives: Set) -> Tuple[List, List, List]:
        # 6 - lemma of left argument,
        # 10 - lemma of right argument
        # 4 - channel name for left argument
        # 8 - channel name for right argument
        train, valid, test = [], [], []
        nns_and_nps_indices = []
        brands_indices = defaultdict(list)
        for index in sorted(positives | negatives):
            brand = None
            if self.dataset.keys[index][4] == 'BRAND_NAME':
                brand = self.dataset.keys[index][6]
            elif self.dataset.keys[index][8] == 'BRAND_NAME':
                brand = self.dataset.keys[index][10]
            else:
                nns_and_nps_indices.append(index)
            if brand:
                brands_indices[brand].append(index)

        n_brand_indices = sum(len(indices) for _, indices in brands_indices.items())

        # split equally starting from the least frequent brands
        counter = 0
        for brand in sorted(brands_indices, key=lambda k: len(brands_indices[k])):
            # if some brand has more than 50% of examples -> add it to train
            if len(brands_indices[brand]) > (0.5 * n_brand_indices):
                counter = 0
            if counter % 3 == 0:
                train.extend(brands_indices[brand])
            elif counter % 3 == 1:
                valid.extend(brands_indices[brand])
            elif counter % 3 == 2:
                test.extend(brands_indices[brand])
            counter += 1

        # use held_out indices of type N-N and N-P and split them
        # to make our data sets more like 3:1:1
        train_indices, valid_indices, test_indices = self._split(nns_and_nps_indices)
        train.extend(train_indices)
        valid.extend(valid_indices)
        test.extend(test_indices)
        return train, valid, test

    def generate_datasets(self, balanced: bool, lexical_split: bool, in_domain: str, out_domain: str = None):
        if in_domain:
            indices = [index for index, descriptor in self.dataset.keys.items() if descriptor[0] == in_domain]
        elif out_domain:
            raise NotImplementedError(f'Out domain dataset split not implemented.')
        else:
            indices = self.dataset.keys.keys()

        return self._generate(indices, balanced, lexical_split)


class BaseSampler(data.Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_loaders(data_dir: str,
                keys_file: str,
                vectors_files: List[str],
                batch_size: int,
                balanced: bool = False,
                lexical_split: bool = False,
                in_domain: str = None,
                out_domain: str = None,
                random_seed: int = 42,
                num_workers: int = 0,
                pin_memory: bool = False):
    dataset = BrandProductDataset(
        keys_file=f'{data_dir}/{keys_file}',
        vectors_files=[f'{data_dir}/{file}' for file in vectors_files],
    )

    ds_generator = DatasetGenerator(dataset, random_seed)
    train_indices, valid_indices, test_indices = ds_generator.generate_datasets(balanced, lexical_split, in_domain)

    train_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=BaseSampler(train_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=BaseSampler(valid_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=BaseSampler(test_indices),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader, dataset.vector_size
