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
        return '\t'.join((self.document_id, self.source, self.dest))

    class Element:
        def __init__(self, sent_id: int, lemma: str, channel: str, indices: List[int], context: List[str], ne: bool):
            self.sent_id = sent_id
            self.lemma = lemma
            self.channel = channel
            self.indices = indices
            self.context = context
            self.ne = ne

        @property
        def start_idx(self):
            return self.indices[0]

        def __str__(self):
            return '\t'.join((self.sent_id, self.lemma, self.channel, self.ne, self.indices, self.context))


class Vector:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __str__(self):
        return np.array2string(self.value, separator=', ').replace('\n', '')
