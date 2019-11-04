import numpy as np


def is_valid_row(row):
    return len(row) < 11


def load_batches(datapath, batch_size=10, vectors_engine=None):
    with open(datapath, encoding="utf-8") as in_file:
        dataset = []
        batch = []
        for idx, line in enumerate(in_file, 1):
            row = line.strip().split('\t')

            if is_valid_row(row):
                cls = row[0]
                v1 = np.array(eval(row[1]))
                v2 = np.array(eval(row[2]))
                vectors = [v1, v2]

                if vectors_engine:
                    vc1, vc2 = vectors_engine.make_vectors(row)
                    vectors.append(vc1)
                    vectors.append(vc2)

                batch.append((cls, np.concatenate(vectors)))

                if (idx % batch_size) == 0:
                    dataset.append(batch)
                    batch = []

        if batch:
            dataset.append(batch)

        return dataset


def labels2idx(labels):
    mapping = {
        'no_relation': 0,
        'in_relation': 1,
    }
    return [mapping[label] for label in labels if label in mapping]
