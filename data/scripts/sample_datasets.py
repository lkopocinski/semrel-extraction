import argparse
import glob
import math
import random
from collections import defaultdict

from parse_utils import Relation

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--positive_size', required=True, type=int,
                        help='How many positive examples should be selected.')
    parser.add_argument('--negative_size', required=True, type=int,
                        help='How many negative examples should be selected.')
    parser.add_argument('--source_dir', required=True, help='Path to datasets in context format.')
    parser.add_argument('--output_dir', required=True, help='Save directory path.')

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def load_file(path):
    with open(path, 'r', encoding='utf-8') as in_file:
        return [line.strip() for line in in_file]


def save_lines(lines, path):
    with open(path, 'w', encoding='utf-8') as out_file:
        for line in lines:
            out_file.write(f'{line}\n')


def get_file_name(file_path):
    file_name = file_path.split('/')[-1]
    file_name = file_name.split('.', 1)[0]
    return file_name


def select(path, size):
    files = glob.glob(path)
    files_number = len(files)
    sample_size = math.floor(size / (1 if files_number == 0 else files_number))

    for file_path in files:
        lines = load_file(file_path)
        if len(lines) > sample_size:
            lines = random.sample(lines, sample_size)

        file_name = get_file_name(file_path)
        file_name = f'{file_name}.sampled'

        yield lines, file_name


def select_positive(source_path, size):
    path = f'{source_path}/positive/*.context'
    return select(path, size)


def select_substituted(source_path, size):
    path = f'{source_path}/negative/substituted/*.context'
    return select(path, size)


def select_negative(source_path, size):
    path = f'{source_path}/negative/*.context'
    files = glob.glob(path)
    files_number = len(files)
    sample_size = math.floor(size / (1 if files_number == 0 else files_number))

    for file_path in files:
        lines = load_file(file_path)

        cat_dict = defaultdict(list)
        for line in lines:
            relation = Relation(line)
            if relation.source.channel == '' and relation.dest.channel == '':
                cat_dict['plain'].append(f'{relation}')
            elif relation.source.channel == 'BRAND_NAME' and relation.dest.channel == '':
                cat_dict['brand'].append(f'{relation}')
            elif relation.source.channel == '' and relation.dest.channel == 'PRODUCT_NAME':
                cat_dict['product'].append(f'{relation}')

        out_lines = []
        for key, lines in cat_dict.items():
            if len(lines) > sample_size:
                lines = random.sample(lines, sample_size)
            out_lines.extend(lines)

        file_name = get_file_name(file_path)
        file_name = f'{file_name}.sampled'

        yield out_lines, file_name


def main(argv=None):
    args = get_args(argv)

    source_dir = args.source_dir
    output_dir = args.output_dir

    positive_size = args.positive_size
    negative_size = args.positive_size

    for lines, file_name in select_positive(source_dir, positive_size):
        save_path = f'{output_dir}/positive/{file_name}'
        save_lines(lines, save_path)

    batch_size = math.floor(negative_size / 4)
    for lines, file_name in select_negative(source_dir, batch_size):
        save_path = f'{output_dir}/negative/{file_name}'
        save_lines(lines, save_path)

    for lines, file_name in select_substituted(source_dir, batch_size):
        save_path = f'{output_dir}/negative/substituted/{file_name}'
        save_lines(lines, save_path)


if __name__ == '__main__':
    main()
