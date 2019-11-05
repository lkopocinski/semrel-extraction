import argparse
import glob
import os
import random

import argcomplete


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with corpora and relations files.')
    parser.add_argument('--output-path', required=True, help='Directory to save generated splits.')
    parser.add_argument('--directories', nargs='+', required=True, help='Directories names to process.')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    for directory in args.dirs:
        path = os.path.join(args.data_in, directory)
        if os.path.isdir(path):
            train, valid, test = split(path)
            save_list(f'{args.output_path}/train/', f'{directory}.list', train)
            save_list(f'{args.output_path}/valid/', f'{directory}.list', valid)
            save_list(f'{args.output_path}/test/', f'{directory}.list', test)


def split(dir_path):
    path = f'{dir_path}/*.rel.xml'
    files = glob.glob(path)
    random.shuffle(files)
    return chunk(files)


def chunk(seq):
    avg = len(seq) / float(5)
    t_len = int(3 * avg)
    v_len = int(avg)
    return [seq[0:t_len], seq[t_len:t_len + v_len], seq[t_len + v_len:]]


def save_list(path, file_name, files_list):
    try:
        os.mkdir(path)
    except OSError:
        print(f'List saving filed. Can not create {path} directory.')
    else:
        file_path = os.path.join(path, file_name)
        with open(file_path, 'w', encoding='utf-8') as out_file:
            for line in files_list:
                out_file.write(f'{line}\n')


if __name__ == '__main__':
    main()
