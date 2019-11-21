from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
from corpus_ccl import token_utils as tou

from data.scripts.utils.corpus import corpora_files, load_document, id_to_sent_dict, \
    is_ner_relation, is_in_channel, get_relation_element

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


def distance_hist(list_file, channels, nr):
    to_save = []
    for corpora_file, relations_file in corpora_files(list_file):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_element = get_relation_element(f, sentences)
                    t_element = get_relation_element(t, sentences)

                    if f_element and t_element:
                        rel = Relation(f_element, t_element)
                        if rel.source.context == rel.dest.context:
                            if rel.dest.start_idx > rel.source.start_idx:
                                shift = (len(rel.source.indices) - 1) if len(rel.source.indices) > 1 else 0
                                dist = rel.dest.start_idx - (rel.source.start_idx + shift)
                                dist = abs(dist)
                            elif rel.source.start_idx > rel.dest.start_idx:
                                shift = (len(rel.dest.indices) - 1) if len(rel.dest.indices) > 1 else 0
                                dist = rel.source.start_idx - (rel.dest.start_idx + shift)
                                dist = abs(dist)
                            else:
                                dist = 0
                            to_save.append(dist)

    hist = pd.Series(to_save)
    hist = hist.value_counts()
    hist = hist.sort_index()
    hist.to_csv(f'{nr}.dist.hist', sep='\t', header=False)


def same_brand(list_file, channels, nr):
    sizes = []

    for corpora_file, relations_file in corpora_files(list_file):
        brand_dict = defaultdict(int)

        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_element = get_relation_element(f, sentences)
                    t_element = get_relation_element(t, sentences)

                    if f_element and t_element:
                        if f_element.channel == "BRAND_NAME":
                            brand_dict[f_element.lemma] += 1
                        elif t_element.channel == "BRAND_NAME":
                            brand_dict[t_element.lemma] += 1

        brands = set(brand_dict.keys())
        size = len(brands)
        print(Path(corpora_file).stem.replace('.ne', ''), size, brands)
        sizes.append(size)

    hist = pd.Series(sizes)
    hist = hist.value_counts()
    hist = hist.sort_index()
    hist.to_csv(f'{nr}.multi.brands.hist', sep='\t', header=False)


def appear_in_text(list_file, channels, nr):
    sizes = []

    for corpora_file, relations_file in corpora_files(list_file):
        brand_dict = defaultdict(int)

        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        token_orths = []
        for par in document.paragraphs():
            for sentence in par.sentences():
                for token in sentence.tokens():
                    if tou.get_annotation(sentence, token, "BRAND_NAME", default=0) == 0 and tou.get_annotation(
                            sentence, token, "BRAND_NAME_IMP", default=0) == 0:
                        ann_number = tou.get_annotation(sentence, token, "PRODUCT_NAME", default=0)
                        token_orths.append((ann_number, token.orth_utf8()))

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_element = get_relation_element(f, sentences)
                    t_element = get_relation_element(t, sentences)

                    if f_element and t_element:
                        if f_element.channel == "PRODUCT_NAME":
                            for nr, orth in token_orths:
                                if f.annotation_number() != nr:
                                    for idx in f_element.indices:
                                        if f_element.context[idx] == orth:
                                            brand_dict[f_element.context[idx]] += 1
                        if t_element.channel == "PRODUCT_NAME":
                            for nr, orth in token_orths:
                                if t.annotation_number() != nr:
                                    for idx in t_element.indices:
                                        if t_element.context[idx] == orth:
                                            brand_dict[t_element.context[idx]] += 1

        brands = set(brand_dict.keys())
        size = len(brands)
        print(Path(corpora_file).stem.replace('.ne', ''), size, brands)
        sizes.append(size)

    hist = pd.Series(sizes)
    hist = hist.value_counts()
    hist = hist.sort_index()
    hist.to_csv(f'{nr}.inclusiv.hist', sep='\t', header=False)


if __name__ == '__main__':
    appear_in_text('81.files', ["BRAND_NAME", "PRODUCT_NAME"], 81)
