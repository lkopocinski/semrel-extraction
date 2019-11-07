#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path

import argcomplete

from relation import Relation
from utils import corpora_files, load_document, id_to_sent_dict, \
    is_ner_relation, is_in_channel, get_relation_element, \
    get_relation_element_multiword


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with split lists of files.')
    parser.add_argument('--output-path', required=True, help='Directory to save generated datasets.')
    parser.add_argument('--multiword', type=bool, default=False, required=False,
                        help='Should generate in multiword mode or not')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    for set_name in ['train', 'valid', 'test']:
        source_dir = os.path.join(args.data_in, set_name, 'positive')
        for list_file in glob.glob(f'{source_dir}/*.list'):
            file_path = os.path.join(args.output_path, 'positive', get_file_name(list_file))
            with open(file_path, 'w', encoding='utf-8') as out_file:
                for example in generate(list_file, ('BRAND_NAME', 'PRODUCT_NAME')):
                    out_file.write(f'{example}\n')


def get_file_name(file_path):
    return Path(file_path).stem


def generate(list_file, channels, multiword=False):
    for corpora_file, relations_file in corpora_files(list_file):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    if multiword:
                        f_element = get_relation_element_multiword(f, sentences)
                        t_element = get_relation_element_multiword(t, sentences)
                    else:
                        f_element = get_relation_element(f, sentences)
                        t_element = get_relation_element(t, sentences)

                    if f_element.start_idx != -1 and t_element.start_idx != -1:
                        yield Relation(f_element, t_element)


if __name__ == "__main__":
    main()
