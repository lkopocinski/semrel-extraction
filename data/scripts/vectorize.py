#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import argcomplete
import numpy as np

from utils.io import save_lines
from utils.embedd import ElmoEmb
from models import Relation

np.set_printoptions(threshold=sys.maxsize)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with sampled datasets.')
    parser.add_argument('--output-path', required=True, help='Directory to save vectors.')
    parser.add_argument('--weights', required=True, help="File with weights to elmo model.")
    parser.add_argument('--options', required=True, help="File with options to elmo model.")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    elmo = ElmoEmb(args.options, args.weights)
    elmo_conv = ElmoEmb(args.options, args.weights, 'convolution')

    for set_name in ['train', 'valid', 'test']:
        for label_type, label_name in zip(['positive', 'negative'], ['in_relation', 'no_relation']):
            source_path = Path(f'{args.data_in}/{set_name}/{label_type}')
            if source_path.is_dir():
                for file_path in source_path.glob('*.sampled'):
                    out_file_path = Path(f'{args.output_path}/{set_name}/{label_type}/{file_path.stem}.vectors')
                    lines = create_vectors(elmo, elmo_conv, file_path, label_name)
                    save_lines(out_file_path, lines)


def create_vectors(elmo, elmo_conv, path, relation_type):
    with open(path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            relation = Relation.from_line(line)

            vector_from = elmo.embedd(relation.source)
            vector_to = elmo.embedd(relation.dest)

            vector_conv_from = elmo_conv.embedd(relation.source)
            vector_conv_to = elmo_conv.embedd(relation.dest)

            yield f'{relation_type}\t{vector_from}\t{vector_to}\t{relation.source}\t{relation.dest}\t{vector_conv_from}\t{vector_conv_to}'


if __name__ == "__main__":
    main()
