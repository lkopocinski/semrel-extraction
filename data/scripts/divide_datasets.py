import argparse
import glob
import math
import random

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
    parser.add_argument('--output_dir', required=True, help='Path to datasets in context format.')

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


def select_positive(source_path, size):
    positive_path = f'{source_path}/positive/*.context'
    files = glob.glob(positive_path)
    files_number = len(files)
    sample_size = math.floor(size / (1 if files_number == 0 else files_number))

    for file_path in files:
        lines = load_file(file_path)
        if len(lines) > sample_size:
            lines = random.sample(lines, sample_size)

        file_name = get_file_name(file_path)
        file_name = f'{file_name}.sampled'

        yield lines, file_name


def main(argv=None):
    args = get_args(argv)

    source_dir = args.source_dir
    output_dir = args.output_dir

    positive_size = args.positive_size
    negative_size = args.positive_size

    for lines, file_name in select_positive(source_dir, positive_size):
        save_path = f'{output_dir}/positive/{file_name}'
        save_lines(lines, save_path)


if __name__ == '__main__':
    main()
