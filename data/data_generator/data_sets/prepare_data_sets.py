import argparse
import math
import random
from collections import defaultdict

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', required=True,
                        help="A file to be processed")
    parser.add_argument('-n', '--save_file_name', required=True,
                        help="Save file name")
    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def get_brand_products_dict(path):
    brand_product_dict = defaultdict(list)
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            idx_brand, ctx_brand, idx_product, ctx_product = extract_relation(line)
            brand = ctx_brand[idx_brand]
            brand_product_dict[brand].append((idx_brand, ctx_brand, idx_product, ctx_product))
    return brand_product_dict


def extract_relation(line):
    line = line.strip()
    by_tab = line.split('\t')

    idx_from, ctx_from = by_tab[0].split(':', 1)
    idx_to, ctx_to = by_tab[1].split(':', 1)

    ctx_from = eval(ctx_from)
    ctx_to = eval(ctx_to)

    idx_from = int(idx_from)
    idx_to = int(idx_to)

    return idx_from, ctx_from, idx_to, ctx_to


def main(argv=None):
    args = get_args(argv)
    brand_products_dict = get_brand_products_dict(args.source_file)
    for brand, lines in brand_products_dict.items():
        size = len(lines)
        if size >= 5:
            size_valid = math.floor(size * 0.2)
            size_test = math.floor(size * 0.2)
            train, valid, test = sample_sets(lines, size_valid, size_test)
        else:
            if size == 4:
                train, valid, test = sample_sets(lines, size_valid=1, size_test=2)
            elif size == 3:
                train, valid, test = sample_sets(lines, size_valid=1, size_test=1)
            elif size == 2:
                train, valid, test = sample_sets(lines, size_valid=0, size_test=1)
            elif size == 1:
                train, valid, test = sample_sets(lines, size_valid=0, size_test=0)

        save_to_file(train, f'train.{args.save_file_name}.context')
        save_to_file(valid, f'valid.{args.save_file_name}.context')
        save_to_file(test, f'test.{args.save_file_name}.context')


def sample_sets(data, size_valid, size_test):
    valid = random.sample(data, size_valid)
    data = [line for line in data if line not in valid]
    test = random.sample(data, size_test)
    data = [line for line in data if line not in test]
    train = data
    return train, valid, test


def save_to_file(data, file_name):
    with open(file_name, mode='a', encoding='utf8') as f:
        lines = ['{}:{}\t{}:{}\n'.format(*line) for line in data]
        f.writelines(lines)


if __name__ == '__main__':
    main()
