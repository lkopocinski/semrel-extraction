#!/usr/bin/env python

import argparse
import sys
from collections import defaultdict

import numpy as np

from utils import print_element

np.set_printoptions(threshold=sys.maxsize)

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', required=True, help="A file with relations contexts.")
    parser.add_argument('-t', '--type', required=True, help="A type of substitution one of: ('brand', 'product')")

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def get_brand_products_dict(path):
    brand_product_dict = defaultdict(list)
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            idx_brand, ctx_brand, idx_product, ctx_product = extract_relation(line)
            brand = ctx_brand[idx_brand]
            product = ctx_product[idx_product]
            brand_product_dict[brand].append(product)
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


def substitute(path, sub_type, brand_products_dict):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            idx_brand, ctx_brand, idx_product, ctx_product = extract_relation(line)
            from_brand = ctx_brand[idx_brand]
            to_product = ctx_product[idx_product]

            for brand, products in brand_products_dict.items():
                if brand.lower() != from_brand.lower():
                    if sub_type == 'brand':
                        substituted = substitute_brand(brand, idx_brand, ctx_brand, idx_product, ctx_product)
                        print_element(*substituted)
                    elif sub_type == 'product':
                        for product in products:
                            substituted = substitute_product(product, idx_brand, ctx_brand, idx_product, ctx_product)
                            print_element(*substituted)


def substitute_brand(brand, idx_brand, ctx_brand, idx_product, ctx_product):
    ctx_brand[idx_brand:idx_brand + 1] = brand.split(' ')
    ctx_product[idx_product:idx_product + 1] = ctx_product[idx_product].split(' ')
    return idx_brand, ctx_brand, idx_product, ctx_product


def substitute_product(product, idx_brand, ctx_brand, idx_product, ctx_product):
    ctx_brand[idx_brand:idx_brand + 1] = ctx_brand[idx_brand].split(' ')
    ctx_product[idx_product:idx_product + 1] = product.split(' ')
    return idx_brand, ctx_brand, idx_product, ctx_product


def main(argv=None):
    args = get_args(argv)
    brand_products_dict = get_brand_products_dict(args.source_file)
    substitute(args.source_file, args.type, brand_products_dict)


if __name__ == "__main__":
    main()
