#!/usr/bin/env python3

import argparse
from pathlib import Path

import argcomplete
from generator import generate_positive, generate_negative

from .utils.io import save_lines


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with lists of files split into datasets.')
    parser.add_argument('--output-path', required=True, help='Directory Directory for saving generated datasets.')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    for set_name in ['train', 'valid', 'test']:
        source_path = Path(f'{args.data_in}/{set_name}')
        if source_path.is_dir():
            for list_file in source_path.glob('*.list'):
                for label_type, gen in zip(['positive', 'negative'], [generate_positive, generate_negative]):
                    out_file_path = Path(f'{args.output_path}/{set_name}/{label_type}/{list_file.stem}.context')
                    lines = gen(list_file, ('BRAND_NAME', 'PRODUCT_NAME'))
                    save_lines(out_file_path, lines)


if __name__ == "__main__":
    main()
