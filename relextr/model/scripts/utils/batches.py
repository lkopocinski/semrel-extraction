from functools import reduce

import numpy as np

from relextr.model.scripts.model.models import RelationVec


class Dataset:

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


class BatchLoader:

    def __init__(self, batch_size, vectors_engine=None):
        self.batch_size = batch_size
        self.vectors_engine = vectors_engine

    def load(self, datapath):
        with open(datapath, encoding="utf-8") as in_file:
            dataset = Dataset()
            batch = []
            for idx, line in enumerate(in_file, 1):
                try:
                    relation = RelationVec(line)
                    vectors = [relation.source.vector, relation.dest.vector]

                    if self.vectors_engine:
                        vc1, vc2 = self.vectors_engine.make_vectors(relation)
                        vectors.append(vc1)
                        vectors.append(vc2)

                    batch.append((relation.label, np.concatenate(vectors)))

                    if (idx % self.batch_size) == 0:
                        dataset.add(batch)
                        batch = []
                except Exception:
                    continue

            if batch:
                dataset.add(batch)

            return dataset
