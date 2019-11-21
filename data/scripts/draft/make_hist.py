from pathlib import Path
from typing import List
import pandas as pd
import math

root_path = '../../generations'
paths = ['test', 'train', 'valid']
nrs = [81, 82, 83]


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

        source = cls.Element(lemma_source, channel_source, [int(index_source)],
                             eval(context_source), float(ne_source))
        dest = cls.Element(lemma_dest, channel_dest, [int(index_dest)],
                           eval(context_dest), float(ne_dest))
        return cls(source, dest)

    def __str__(self):
        return f'{self.source.lemma} : {self.dest.lemma}' \
            f'\t{self.source.channel} : {self.dest.channel}' \
            f'\t{self.source.start_idx}:{self.source.context}' \
            f'\t{self.dest.start_idx}:{self.dest.context}' \
            f'\t{self.source.ne}:{self.dest.ne}'

    class Element:
        def __init__(self, lemma: str, channel: str, indices: List[int], context: List[str], ne: int):
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


def brand_hist():
    for nr in nrs:
        to_save = []
        for path in paths:
            df = pd.read_csv(f'{root_path}/{path}/positive/{nr}.context', sep='\t| : ', engine='python')
            brands = df[df.iloc[:, 2] == "BRAND_NAME"].iloc[:, 0]
            to_save.append(brands)

            brands = df[df.iloc[:, 3] == "BRAND_NAME"].iloc[:, 1]
            to_save.append(brands)

        hist = pd.concat(to_save)
        hist = hist.value_counts()
        hist.to_csv(f'{nr}.hist', sep='\t', header=False)


def distance_hist():
    for nr in nrs:
        to_save = []
        for path in paths:
            file_path = Path(f'{root_path}/{path}/positive/{nr}.context')
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    rel = Relation.from_line(line)
                    if rel.source.context == rel.dest.context:
                        if rel.dest.start_idx > rel.source.start_idx:
                            dist = rel.dest.start_idx - (rel.source.start_idx + (len(rel.source.indices) - 1))
                        else:
                            dist = rel.source.start_idx - (rel.dest.start_idx + (len(rel.dest.indices) - 1))
                        to_save.append(dist)

        hist = pd.Series(to_save)
        hist = hist.value_counts()
        hist = hist.sort_index()
        hist.to_csv(f'{nr}.hist', sep='\t', header=False)


if __name__ == '__main__':
    distance_hist()
