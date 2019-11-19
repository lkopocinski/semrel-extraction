#!/usr/bin/env python3

import argparse
import collections
import os
from pathlib import Path

from corpus_ccl import cclutils

from utils.embedd import ElmoEmb

EmbeddArgs = collections.namedtuple('EmbeddArgs', 'start_idx context')


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


def create_map(list_file, elmo):
    for document in load_documents(list_file):
        for paragraph in document.paragraphs():
            for sentence in paragraph.sentences():
                id_doc = get_doc_id(document)
                id_sent = sentence.id()
                context = [token.orth_utf8() for token in sentence.tokens()]

                for id_tok, token in enumerate(sentence.tokens()):
                    orth = token.orth_utf8()
                    vector = elmo.embedd(EmbeddArgs(id_tok, context))
                    print(f'{id_doc}\t{id_sent}\t{id_tok}\t{orth}\t{vector}')


def main(argv=None):
    args = get_args(argv)
    elmo = ElmoEmb(args.options, args.weights)
    create_map(args.indexfiles, elmo)


if __name__ == '__main__':
    main()
