#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path
import argcomplete

from generator import generate_positive, generate_negative


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
        source_dir = os.path.join(args.data_in, set_name)

        for list_file in glob.glob(f'{source_dir}/*list'):
            file_name = f'{get_file_name(list_file)}.context'

            file_path = os.path.join(args.output_path, set_name, 'positive')
            lines = generate_positive(list_file, ('BRAND_NAME', 'PRODUCT_NAME'))
            save_lines(file_path, file_name, lines)

            file_path = os.path.join(args.output_path, set_name, 'negative')
            lines = generate_negative(list_file, ('BRAND_NAME', 'PRODUCT_NAME'))
            save_lines(file_path, file_name, lines)


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


if __name__ == "__main__":
    main()
