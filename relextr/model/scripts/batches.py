import numpy as np

from utils import Relation


class BatchLoader:

    def __init__(self, batch_size, vectors_engine=None):
        self.batch_size = batch_size
        self.vectors_engine = vectors_engine

    def load(self, datapath):
        with open(datapath, encoding="utf-8") as in_file:
            dataset = []
            batch = []
            for idx, line in enumerate(in_file, 1):
                try:
                    relation = Relation(line)
                    vectors = [relation.source.vector, relation.dest.vector]

                    if self.vectors_engine:
                        vc1, vc2 = self.vectors_engine.make_vectors(relation)
                        vectors.append(vc1)
                        vectors.append(vc2)

                    batch.append((relation.label, np.concatenate(vectors)))

                    if (idx % self.batch_size) == 0:
                        dataset.append(batch)
                        batch = []
                except Exception:
                    continue

            if batch:
                dataset.append(batch)

            return dataset
