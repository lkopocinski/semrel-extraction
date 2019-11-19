from typing import List

import numpy as np


class Relation:

    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

    @classmethod
    def from_line(cls, line: str):
        row = line.strip().split('\t')

        lemma_source, lemma_dest = row[0].replace(' : ', ':').split(':', 1)
        channel_source, channel_dest = row[1].replace(' : ', ':').split(':', 1)
        index_source, context_source = row[2].split(':', 1)
        index_dest, context_dest = row[3].split(':', 1)
        ne_source, ne_dest = row[4].split(':', 1)

        source = cls.Element(lemma_source, channel_source, [int(index_source)], eval(context_source), ne_source)
        dest = cls.Element(lemma_dest, channel_dest, [int(index_dest)], eval(context_dest), ne_dest)
        return cls(source, dest)

    def __str__(self):
        return f'{self.source.lemma} : {self.dest.lemma}' \
            f'\t{self.source.channel} : {self.dest.channel}' \
            f'\t{self.source.start_idx}:{self.source.context}' \
            f'\t{self.dest.start_idx}:{self.dest.context}' \
            f'\t{self.dest.ne}:{self.dest.ne}'

    class Element:
        def __init__(self, lemma: str, channel: str, indices: List[int], context: List[str], ne: bool):
            self.lemma = lemma
            self.channel = channel
            self.indices = indices
            self.context = context
            self.ne = ne

        @property
        def start_idx(self):
            return self.indices[0]

        def __str__(self):
            return f'{self.lemma}\t{self.channel}\t{self.start_idx}:{self.context}'


class Vector:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return np.array2string(self.value, separator=', ').replace('\n', '')
