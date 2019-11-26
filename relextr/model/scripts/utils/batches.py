from functools import reduce
from pathlib import Path
from typing import List

import numpy as np
from torch.utils import data

from relextr.model.scripts.model.models import PairVec


class DatasetOld:

    def __init__(self):
        self.batches = []

    def add(self, batch):
        self.batches.append(batch)

    @property
    def vector_size(self):
        return len(self.batches[0][0][1])

    @property
    def size(self):
        return reduce((lambda x, y: x + len(y)), self.batches, 0)


class Dataset(data.dataset):
    label2digit = {
        'no_relation': 0,
        'in_relation': 1,
    }

    @staticmethod
    def from_file(path: Path, methods: List):
        with path.open('r', encoding="utf-8") as f:
            lines = [line for line in f]
            return Dataset(lines, methods)

    def __init__(self, lines: List[str], methods: List):
        self.lines = lines
        self.methods = methods

    @property
    def vector_size(self):
        x, _ = self[0]
        return len(x)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index]
        pair = PairVec(line)

        X = pair.make_vector(self.methods)
        y = self.label2digit[pair.label]

        return X, y


class BatchLoader:

    def __init__(self, batch_size, vectors_engines=None):
        self.batch_size = batch_size
        self.vectors_engines = vectors_engines

    def load(self, datapath):
        with open(datapath, encoding="utf-8") as in_file:
            dataset = Dataset()
            batch = []
            for idx, line in enumerate(in_file, 1):
                try:
                    relation = PairVec(line)
                    vectors = [relation.source.vector, relation.dest.vector]

                    for engine in self.vectors_engines:
                        if engine:
                            vc1, vc2 = engine.make_vectors(relation)
                            vectors.append(vc1)
                            vectors.append(vc2)

                    batch.append((relation.label, np.concatenate(vectors)))

                    if (idx % self.batch_size) == 0:
                        dataset.add(batch)
                        batch = []
                except Exception as e:
                    print(e)
                    continue

            if batch:
                dataset.add(batch)

            return dataset
