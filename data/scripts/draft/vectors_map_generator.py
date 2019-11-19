#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from corpus_ccl import cclutils


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--indexfiles', required=True, help='File with corpus documents paths.')
    return parser.parse_args(argv)


def load_documents(fileindex):
    with open(fileindex, 'r', encoding='utf-8') as f:
        for line in f:
            filepath = line.strip()
            if not os.path.exists(filepath):
                continue
            cclpath = filepath.replace('rel.xml', '.xml')
            relpath = filepath
            yield cclutils.read_ccl_and_rel_ccl(cclpath, relpath)


def get_doc_id(document):
    ccl_path, rel_path = document.path().split(';')
    return Path(ccl_path).stem


def create_map(list_file):
    for document in load_documents(list_file):
        for paragraph in document.paragraphs():
            for sentence in paragraph.sentences():
                id_doc = get_doc_id(document)
                id_sent = sentence.id()

                for id_tok, token in enumerate(sentence.tokens()):
                    orth = token.orth_utf8()
                    print(f'{id_doc}\t{id_sent}\t{id_tok}\t{orth}\t[vector]')


def main(argv=None):
    args = get_args(argv)
    create_map(args.indexfiles)


if __name__ == '__main__':
    main()
