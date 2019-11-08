#!/usr/bin/env python3

import argparse
import glob
import os
import random

import argcomplete


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
        path = os.path.join(args.data_in, directory)
        if os.path.isdir(path):
            train_list, valid_list, test_list = split(path)
            save_lines(
                file_path=os.path.join(args.output_path, 'train', f'{directory}.list'),
                lines=train_list)
            save_lines(
                file_path=os.path.join(args.output_path, 'valid', f'{directory}.list'),
                lines=valid_list)
            save_lines(
                file_path=os.path.join(args.output_path, 'test', f'{directory}.list'),
                lines=test_list)


def split(dir_path):
    files = glob.glob(f'{dir_path}/*.rel.xml')
    random.shuffle(files)
    return chunk(files)


def chunk(seq):
    avg = len(seq) / float(5)
    t_len = int(3 * avg)
    v_len = int(avg)
    return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]


def save_lines(file_path, lines):
    directory = os.path.dirname(file_path)
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(f'Saving filed. Can not create {directory} directory.')
    else:
        with open(file_path, 'w', encoding='utf-8') as out_file:
            for line in lines:
                out_file.write(f'{line}\n')


if __name__ == '__main__':
    main()
