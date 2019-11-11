#!/usr/bin/env python3

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import argcomplete
from model.relation import Relation


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with context format files.')
    parser.add_argument('--output-path', required=True, help='Directory for saving sampled datasets.')
    parser.add_argument('--train-size', nargs=2, type=int, required=True,
                        help='Train dataset batch sizes [positive, negative]')
    parser.add_argument('--valid-size', nargs=2, type=int, required=True,
                        help='Validation dataset batch sizes [positive, negative]')
    parser.add_argument('--test-size', nargs=2, type=int, required=True,
                        help='Test dataset batch sizes [positive, negative]')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    for set_name, size in zip(['train', 'valid', 'test'], [args.train_size, args.valid_size, args.test_size]):
        source_dir = Path(f'{args.data_in}/{set_name}')
        output_dir = Path(f'{args.output_path}/{set_name}')
        pos_batch_size, neg_batch_size = size

        for lines, file_name in select_positive(source_dir, pos_batch_size):
            save_path = os.path.join(output_dir, 'positive')
            save_lines(save_path, file_name, lines)

        for lines, file_name in select_negative(source_dir, neg_batch_size):
            save_path = os.path.join(output_dir, 'negative')
            save_lines(save_path, file_name, lines)


def select_positive(source_path, batch_size):
    return select(
        path=f'{source_path}/positive/*.context',
        size=batch_size
    )


def select(path, size):
    for file_path in glob.glob(path):
        lines = load_file(file_path)
        if len(lines) > size:
            lines = random.sample(lines, size)

        file_name = f'{get_file_name(file_path)}.sampled'
        yield lines, file_name


def load_file(path):
    with open(path, 'r', encoding='utf-8') as in_file:
        return [line.strip() for line in in_file]


def select_negative(source_path, size):
    size = math.floor(size / 3)
    path = f'{source_path}/negative/*.context'

    for file_path in glob.glob(path):
        lines = load_file(file_path)

        type_dict = defaultdict(list)
        for line in lines:
            relation = Relation(line)
            if relation.source.channel == '' and relation.dest.channel == '':
                type_dict['plain'].append(f'{relation}')
            elif relation.source.channel == 'BRAND_NAME' and relation.dest.channel == '':
                type_dict['brand'].append(f'{relation}')
            elif relation.source.channel == '' and relation.dest.channel == 'PRODUCT_NAME':
                type_dict['product'].append(f'{relation}')

        out_lines = []
        for key, lines in type_dict.items():
            if len(lines) > size:
                lines = random.sample(lines, size)
            out_lines.extend(lines)

        file_name = f'{get_file_name(file_path)}.sampled'
        yield out_lines, file_name


if __name__ == '__main__':
    main()
