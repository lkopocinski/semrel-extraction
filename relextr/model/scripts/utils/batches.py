from pathlib import Path
from typing import List

import numpy as np
from torch.utils import data


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


class PairVec:

    def __init__(self, line):
        self.line = line
        self._init_from_line()

    def _init_from_line(self):
        row = self.line.strip().split('\t')

        # Details
        self.label = row[0]
        self.lemma1, self.lemma2 = row[1], row[2]
        self.channel1, self.channel2 = row[3], row[4]
        self.ne1, self.ne2 = row[5], row[6]
        self.indices1, self.indices2 = eval(row[7]), eval(row[8])
        self.context1, self.context2 = eval(row[9]), eval(row[10])

        # Vectors
        self.elmo1, self.elmo2 = eval(row[11]), eval(row[12])
        self.elmoconv1, self.elmoconv2 = eval(row[13]), eval(row[14])
        self.fasttext1, self.fasttext2 = eval(row[15]), eval(row[16])
        self.sent2vec1, self.sent2vec2 = eval(row[17]), eval(row[18])
        self.retrofit1, self.retrofit2 = eval(row[19]), eval(row[20])
        self.ner1, self.ner2 = eval(row[21]), eval(row[22])

    @property
    def elmo(self):
        return self.elmo1, self.elmo2

    @property
    def elmoconv(self):
        return self.elmoconv1, self.elmoconv2

    @property
    def fasttext(self):
        return self.fasttext1, self.fasttext2

    @property
    def sent2vec(self):
        return self.sent2vec1, self.sent2vec2

    @property
    def retrofit(self):
        return self.retrofit1, self.retrofit2

    @property
    def ner(self):
        return self.ner1, self.ner2

    def make_vector(self, methods: List[str]):
        vectors = [self.elmo1, self.elmo2]

        for method in methods:
            try:
                v1, v2 = getattr(self, method)
                vectors.extend([v1, v2])
            except AttributeError:
                print(f'There is no method called {method}')

        return np.concatenate(vectors)
