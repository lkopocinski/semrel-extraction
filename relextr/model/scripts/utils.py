from functools import reduce

import numpy as np


def get_set_size(dataset_batches):
    return reduce((lambda x, y: x + len(y)), dataset_batches, 0)


def is_better_fscore(fscore, best_fscore):
    return fscore[0] > best_fscore[0] and fscore[1] > best_fscore[1]


def labels2idx(labels):
    mapping = {
        'no_relation': 0,
        'in_relation': 1,
    }
    return [mapping[label] for label in labels if label in mapping]


def load_batches(datapath, batch_size=10, vectors_engine=None):
    with open(datapath, encoding="utf-8") as in_file:
        dataset = []
        batch = []
        for idx, line in enumerate(in_file, 1):
            try:
                relation = Relation(line)
                vectors = [relation.source.vector, relation.dest.vector]

                if vectors_engine:
                    vc1, vc2 = vectors_engine.make_vectors(relation)
                    vectors.append(vc1)
                    vectors.append(vc2)

                batch.append((relation.label, np.concatenate(vectors)))

                if (idx % batch_size) == 0:
                    dataset.append(batch)
                    batch = []
            except Exception:
                continue

        if batch:
            dataset.append(batch)

        return dataset


class Relation:

    def __init__(self, line):
        self.line = line
        self._from = None
        self._to = None
        self.label = None
        self._init_from_line()

    def _init_from_line(self):
        line = self.line.strip()
        by_tab = line.split('\t')

        self.label = by_tab[0]
        vector_from, vector_to = np.array(eval(by_tab[1])), np.array(eval(by_tab[2]))
        lemma_from, lemma_to = by_tab[3], by_tab[6]
        channel_from, channel_to = by_tab[4], by_tab[7]
        index_from, context_from = by_tab[5].split(':', 1)
        index_to, context_to = by_tab[8].split(':', 1)
        conv_vector_from, conv_vector_to = np.array(eval(by_tab[9])), np.array(eval(by_tab[10]))

        context_from = eval(context_from)
        context_to = eval(context_to)

        index_from = int(index_from)
        index_to = int(index_to)

        self._from = self.Element(vector_from, lemma_from, channel_from, index_from, context_from, conv_vector_from)
        self._to = self.Element(vector_to, lemma_to, channel_to, index_to, context_to, conv_vector_to)

    @property
    def source(self):
        return self._from

    @property
    def dest(self):
        return self._to

    class Element:
        def __init__(self, vector, lemma, channel, index, context, conv_vector):
            self.vector = vector
            self.lemma = lemma
            self.channel = channel
            self.index = index
            self.context = context
            self.conv_vector = conv_vector
