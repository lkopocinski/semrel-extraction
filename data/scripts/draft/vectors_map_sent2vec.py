#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import torch
import sent2vec

import numpy as np
from corpus_ccl import cclutils

np.set_printoptions(threshold=sys.maxsize)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--relations-file', required=True, help='File with generated relations.')
    parser.add_argument('--sent2vec', required=True, help="File with sent2vec model.")
    return parser.parse_args(argv)


def file_rows(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip().split('\t')


def make_sentence_map(path):
    corpus_path = f'/data6/lukaszkopocinski/repos/semrel-extraction/data/corpora/'

    sentence_map = {}
    for domain, _, doc_id, f_sent_id, _, _, _, f_indices, _, t_sent_id, _, _, _, t_indices, _ in file_rows(path):
        if (domain, doc_id) not in sentence_map:
            document = cclutils.read_ccl(f'{corpus_path}/{domain}/{doc_id}.xml')
            sentence_map[(domain, doc_id)] = {
                int(sentence.id().replace('sent', '')): [token.orth_utf8() for token in sentence.tokens()]
                for par in document.paragraphs()
                for sentence in par.sentences()
            }
    return sentence_map


def mask_tokens(tokes, indices):
    return [token if idx in indices else 'MASK' for idx, token in enumerate(tokes)]


def make_vectors(relations_file, s2v):
    rel_map = {}
    sentence_map = make_sentence_map(relations_file)

    for domain, _label, doc_id, f_sent_id, _f_lemma, _f_channel, _, f_indices, _, t_sent_id, _t_lemma, _t_channel, _, t_indices, _ in file_rows(relations_file):
        if len(eval(f_indices)) > 5 or len(eval(t_indices)) > 5:
            continue

        f_sent_id_int = int(f_sent_id.replace('sent', ''))
        t_sent_id_int = int(t_sent_id.replace('sent', ''))

        if f_sent_id_int > t_sent_id_int:
            # swap
            f_sent_id_int = int(t_sent_id.replace('sent', ''))
            t_sent_id_int = int(f_sent_id.replace('sent', ''))
            f_indices_int = eval(t_indices)
            t_indices_int = eval(f_indices)
        else:
            f_indices_int = eval(f_indices)
            t_indices_int = eval(t_indices)

        if f_sent_id_int == t_sent_id_int:
            ctx_left = sentence_map[(domain, doc_id)].get(f_sent_id_int - 1, [])
            ctx_between = mask_tokens(sentence_map[(domain, doc_id)][f_sent_id_int], f_indices_int + t_indices_int)
            ctx_right = sentence_map[(domain, doc_id)].get(t_sent_id_int + 1, [])
        elif (t_sent_id_int - f_sent_id_int) > 0:
            # be sure the indices are swapped
            ctx_between = []
            ctx_left = sentence_map[(domain, doc_id)].get(f_sent_id_int - 1, [])
            ctx_between.extend(mask_tokens(sentence_map[(domain, doc_id)][f_sent_id_int], f_indices_int))

            for i in range(f_sent_id_int + 1, t_sent_id_int):
                ctx_between.append(sentence_map[(domain, doc_id)].get(i, []))

            ctx_between.extend(mask_tokens(sentence_map[(domain, doc_id)][t_sent_id_int], t_indices_int))
            ctx_right = sentence_map[(domain, doc_id)].get(t_sent_id_int + 1, [])

        tokens = ctx_left + ctx_between + ctx_right
        sentence = ' '.join(tokens)

        rel_key = (domain, _label, doc_id, f_sent_id, t_sent_id, _f_channel, _t_channel, f_indices, t_indices, _f_lemma, _t_lemma)
        rel_map[rel_key] = torch.FloatTensor(s2v.embed_sentence(sentence).flatten())

    return rel_map


def main(argv=None):
    args = get_args(argv)

    model = sent2vec.Sent2vecModel()
    model.load_model(args.sent2vec, inference_mode=True)

    rel_map = make_vectors(args.relations_file, model)
    keys, vec = zip(*rel_map.items())
    torch.save(vec, 'sen2vec.rel.pt')

    with open('sen2vec.rel.pt.keys', 'w', encoding='utf-8') as f:
        for key in keys:
            f.write(f'{key}\n')


if __name__ == '__main__':
    main()
