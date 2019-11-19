#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from corpus_ccl import cclutils

np.set_printoptions(threshold=sys.maxsize)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--indexfiles', required=True, help='File with corpus documents paths.')
    parser.add_argument('--weights', required=True, help="File with weights to elmo model.")
    parser.add_argument('--options', required=True, help="File with options to elmo model.")
    return parser.parse_args(argv)


def load_documents(fileindex):
    with open(fileindex, 'r', encoding='utf-8') as f:
        for line in f:
            filepath = line.strip()
            if not os.path.exists(filepath):
                continue
            cclpath = filepath
            relpath = filepath.replace('.xml', '.rel.xml')
            yield cclutils.read_ccl_and_rel_ccl(cclpath, relpath)


def get_doc_id(document):
    ccl_path, rel_path = document.path().split(';')
    return Path(ccl_path).stem


def embedd(idx, context, elmo):
    character_ids = batch_to_ids([context])
    embeddings = elmo(character_ids)
    v = embeddings['elmo_representations'][0].data.numpy()
    value = v[:, idx, :].flatten()
    return np.array2string(value, separator=', ').replace('\n', '')


def create_map(list_file, elmo):
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
    elmo = Elmo(args.options, args.weights, 1, dropout=0)
    create_map(args.indexfiles, elmo)


if __name__ == '__main__':
    main()
