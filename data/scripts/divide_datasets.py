import argparse
import glob
import math
from random import random

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_size', required=True, help='How many positive examples should be selected.')
    parser.add_argument('--path', required=True, help='Path to datasets in context format.')

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def load_file(path):
    with open(path, 'r', encoding='utf-8') as in_file:
        return [line.strip() for line in in_file]


def save_lines(lines, path):
    with open(path, 'w', encoding='utf-8') as out_file:
        out_file.writelines(lines)


def select_positive(path, size):
    files = glob.glob(path)
    files_number = len(files)
    batch_size = math.floor(size / files_number)

    for file_path in files:
        lines = load_file(file_path)
        lines = random.sample(lines, batch_size)

        file = file_path.split('.')[-2]
        yield lines, file


def main(argv=None):
    args = get_args(argv)
    path = args.path
    positive_size = args.positive_size

    positive_path = f'{path}/positive/*.context'
    for lines, file in select_positive(positive_path, positive_size):
        save_lines(lines, path)


if __name__ == '__main__':
    main()
