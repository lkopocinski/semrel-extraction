#!/usr/bin/env python3

import argparse
import glob
import os
import sys
from pathlib import Path

import argcomplete
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids
from parse_utils import Relation

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
    elmo, elmo_conv = init_elmo(args.options, args.weights)

    for set_name in ['train', 'valid', 'test']:
        source_dir = os.path.join(args.data_in, set_name, 'positive')
        for file_path in glob.glob(f'{source_dir}/*.sampled'):
            lines = create_vectors(elmo, elmo_conv, file_path, 'in_relation')
            save_lines(source_dir, f'{get_file_name(file_path)}.context', lines)

        source_dir = os.path.join(args.data_in, set_name, 'negative')
        for file_path in glob.glob(f'{source_dir}/*.sampled'):
            lines = create_vectors(elmo, elmo_conv, file_path, 'no_relation')
            save_lines(source_dir, f'{get_file_name(file_path)}.context', lines)


def get_file_name(file_path):
    return Path(file_path).stem


def save_lines(path, file_name, lines):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f'List saving filed. Can not create {path} directory.')
    else:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'w', encoding='utf-8') as out_file:
            for line in lines:
                out_file.write(f'{line}\n')


def init_elmo(options_file, weights_file):
    elmo = Elmo(options_file, weights_file,
                num_output_representations=1,
                dropout=0)

    elmo_conv = Elmo(options_file, weights_file,
                     num_output_representations=1,
                     dropout=0,
                     scalar_mix_parameters=[1, -9e10, -9e10])

    return elmo, elmo_conv


class Vector:
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return np.array2string(self.value, separator=', ').replace('\n', '')


def create_vectors(elmo, elmo_conv, path, relation_type):
    with open(path, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            relation = Relation(line)

            vector_from = get_context_vector(elmo, relation.source)
            vector_to = get_context_vector(elmo, relation.dest)

            vector_conv_from = get_context_vector(elmo_conv, relation.source)
            vector_conv_to = get_context_vector(elmo_conv, relation.dest)

            yield f'{relation_type}\t{vector_from}\t{vector_to}\t{relation.source}\t{relation.dest}\t{vector_conv_from}\t{vector_conv_to}'


def get_context_vector(model, element):
    character_ids = batch_to_ids([element.context])
    embeddings = model(character_ids)
    v = embeddings['elmo_representations'][0].data.numpy()
    try:
        value = v[:, element.index, :].flatten()
    except:
        print(element, file=sys.stderr)
    return Vector(value)


if __name__ == "__main__":
    main()
