#!/usr/bin/env python3

import argparse
import random

import argcomplete
from utils import save_lines
from pathlib import Path


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with corpora.')
    parser.add_argument('--output-path', required=True, help='Directory for saving generated splits.')
    parser.add_argument('--directories', nargs='+', required=True, help='Directories names to be processed.')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    for directory in args.directories:
        path = Path(f'{args.data_in}/{directory}')
        if path.is_dir():
            for set_type, set_files in zip(['train', 'valid', 'test'], split(path)):
                file_path = Path(f'{args.output_path}/{set_type}/{directory}.list')
                save_lines(file_path, set_files)


def split(dir_path):
    files = list(dir_path.glob('*.rel.xml'))
    random.shuffle(files)
    return chunk(files)


def chunk(seq):
    avg = len(seq) / float(5)
    t_len = int(3 * avg)
    v_len = int(avg)
    return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]


if __name__ == '__main__':
    main()
