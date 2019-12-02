#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from corpus_ccl import cclutils

from corpus import is_ner_relation, is_in_channel, get_annotation_indices

np.set_printoptions(threshold=sys.maxsize)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--relations-file', required=True, help='File with generated relations.')
    parser.add_argument('--sent2vec', required=True, help="File with sent2vec model.")
    return parser.parse_args(argv)


def file_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            yield line.strip().split('\t')


def make_sentence_map(path):
    sentence_map = {}
    for domain, _, doc_id, f_sent_id, _, _, _, f_indices, _, t_sent_id, _, _, _, t_indices, _ in file_rows(path):
        if (domain, doc_id, f_sent_id) not in sentence_map:

            document = cclutils.read_ccl(
                f'/data6/lukaszkopocinski/repos/semrel-extraction/data/corpora/{domain}/{doc_id}.xml')
            for par in document.paragraphs():
                for sentence in par.sentences():
                    sentence_map[(domain, doc_id, f_sent_id)] = [token.orth_utf8() for token in sentence.tokens()]

        elif (domain, doc_id, t_sent_id) not in sentence_map:

            document = cclutils.read_ccl(
                f'/data6/lukaszkopocinski/repos/semrel-extraction/data/corpora/{domain}/{doc_id}.xml')
            for par in document.paragraphs():
                for sentence in par.sentences():
                    sentence_map[(domain, doc_id, t_sent_id)] = [token.orth_utf8() for token in sentence.tokens()]

    return sentence_map


def make_vectors(relations_file, s2v):
    sentence_map = make_sentence_map(relations_file)

    for domain, _, doc_id, f_sent_id, _, _, _, f_indices, _, t_sent_id, _, _, _, t_indices, _ in file_rows(s2v):


        for document in load_documents(list_file):
            sentences = {
                int(sentence.sentence_id().replace('sent', '')): sentence
                for paragraph in document.paragraphs()
                for sentence in paragraph.sentences()
            }
        for relation in document.relations():
            ctx_left, ctx_between, ctx_right = [], [], []
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()
                    f_sent_id = f.sentence_id()
                    t_sent_id = t.sentence_id()

                    f_sent_id_int = int(f_sent_id.replace('sent', ''))
                    t_sent_id_int = int(t_sent_id.replace('sent', ''))

                    f_ann_num = f.annotation_number()
                    t_ann_num = t.annotation_number()

                    f_ann_chan = f.channel()
                    t_ann_chan = t.channel()

                    if f_sent_id_int > t_sent_id_int:
                        # zamienic
                        pass

                    if f_sent_id_int == t_sent_id_int:
                        f_indices = get_annotation_indices(sentences[f_sent_id_int], f_ann_num, f_ann_chan)
                        t_indices = get_annotation_indices(sentences[t_sent_id_int], t_ann_num, t_ann_chan)

                        ctx_left = list(sentences[f_sent_id_int - 1].tokens()) if (
                                                                                          f_sent_id_int - 1) in sentences else []
                        ctx_between.extend(mask_tokens(sentences[f_sent_id_int], f_indices + t_indices))
                        ctx_right = list(sentences[t_sent_id_int + 1].tokens()) if (
                                                                                           t_sent_id_int + 1) in sentences else []
                    elif (t_sent_id_int - f_sent_id_int) > 1:
                        # be sure the indices are swaped
                        f_indices = get_annotation_indices(sentences[f_sent_id_int], f_ann_num, f_ann_chan)
                        t_indices = get_annotation_indices(sentences[t_sent_id_int], t_ann_num, t_ann_chan)

                        ctx_left = list(sentences[f_sent_id_int - 1].tokens()) if (
                                                                                          f_sent_id_int - 1) in sentences else []
                        ctx_between.extend(mask_tokens([f_sent_id_int], f_indices))

                        for i in range(f_sent_id_int + 1, t_sent_id_int):
                            ctx_between.append(sentences[i])

                        ctx_between.append(mask_tokens(sentences[t_sent_id_int], t_indices))
                        ctx_right = list(sentences[t_sent_id_int + 1].tokens()) if (
                                                                                           t_sent_id_int + 1) in sentences else []

            tokens = [token.orth_utf8() for token in ctx_left + ctx_between + ctx_right]
            sent = ' '.join(tokens)
            value = s2v.embed_sentence(sent).flatten()

    for document in load_documents(list_file):
        for paragraph in document.paragraphs():
            for sentence in paragraph.sentences():
                id_doc = get_doc_id(document)
                id_sent = sentence.id()
                context = [token.orth_utf8() for token in sentence.tokens()]

                for id_tok, token in enumerate(sentence.tokens()):
                    orth = token.orth_utf8()
                    vector = embedd(id_tok, context, elmo)
                    print(f'{id_doc}\t{id_sent}\t{id_tok}\t{orth}\t{vector}')


def main(argv=None):
    args = get_args(argv)


if __name__ == '__main__':
    main()
