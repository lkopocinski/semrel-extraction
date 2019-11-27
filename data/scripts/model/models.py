from typing import List

import numpy as np


class Relation:

    def __init__(self, document_id, source, dest):
        self.document_id = document_id
        self.source = source
        self.dest = dest

    @classmethod
    def from_line(cls, line: str):
        row = line.strip().split('\t')

        document_id = row[2]
        source = cls.Element(*row[3:9])
        dest = cls.Element(*row[9:15])

        return cls(document_id, source, dest)

    def __str__(self):
        return f'{self.document_id}\t{self.source}\t{self.dest}'

    class Element:
        def __init__(self, sent_id: str, lemma: str, channel: str, ne: bool, indices: List[int], context: List[str]):
            self.sent_id = sent_id
            self.lemma = lemma
            self.channel = channel
            self.ne = ne
            self.indices = indices
            self.context = context

        @property
        def start_idx(self):
            return self.indices[0]

        def __str__(self):
            return f'{self.sent_id}\t{self.lemma}\t{self.channel}\t{self.ne}\t{self.indices}\t{self.context}'


class Vector:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __str__(self):
        return np.array2string(self.value, separator=', ').replace('\n', '')
