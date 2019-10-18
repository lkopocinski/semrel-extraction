#!/usr/bin/env python

import argparse
import random
from collections import defaultdict

from .utils import print_element

try:
    import argcomplete
except ImportError:
    argcomplete = None

"""
    Script require a file with multi word phrases as one token.
"""


class NotSupportedRelationError(Exception):
    pass


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_file', required=True,
                        help='A file with relations contexts.')
    parser.add_argument('--sample_size', required=True, type=int,
                        help='How many brands should be selected to substitution.')

    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def get_brand_products_dict(path):
    brand_product_dict = defaultdict(list)
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            try:
                lemma_brand, lemma_product, idx_brand, ctx_brand, idx_product, ctx_product = extract_relation(line)
                brand = ctx_brand[idx_brand]
                brand_product_dict[lemma_brand].append(brand)
            except NotSupportedRelationError as e:
                continue
    return brand_product_dict


def extract_relation(line):
    line = line.strip()
    by_tab = line.split('\t')

    lemma_from, lemma_to = by_tab[0].replace(' : ', ':').split(':', 1)
    channel_from, channel_to = by_tab[1].replace(' : ', ':').split(':', 1)
    idx_from, ctx_from = by_tab[2].split(':', 1)
    idx_to, ctx_to = by_tab[3].split(':', 1)

    ctx_from = eval(ctx_from)
    ctx_to = eval(ctx_to)

    idx_from = int(idx_from)
    idx_to = int(idx_to)

    if channel_from == 'BRAND_NAME' and channel_to == 'PRODUCT_NAME':
        return lemma_from, lemma_to, idx_from, ctx_from, idx_to, ctx_to
    elif channel_from == 'PRODUCT_NAME' and channel_to == 'BRAND_NAME':
        return lemma_to, lemma_from, idx_to, ctx_to, idx_from, ctx_from
    else:
        raise NotSupportedRelationError


def substitute(path, brand_products_dict, sample_size):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            try:
                lemma_brand, lemma_product, idx_brand, ctx_brand, idx_product, ctx_product = extract_relation(line)
            except NotSupportedRelationError as e:
                continue

            for lemma, orths in random.sample(brand_products_dict.items(), sample_size):
                if lemma.lower() != lemma_brand.lower():
                    subst_orth = random.sample(orths, 1)[0]
                    idx_brand_, ctx_brand_, idx_product_, ctx_product_ = substitute_brand(subst_orth, idx_brand,
                                                                                          ctx_brand, idx_product,
                                                                                          ctx_product)
                    print_element(
                        lemma, lemma_product,
                        'BRAND_NAME', 'PRODUCT_NAME',
                        idx_brand_, ctx_brand_,
                        idx_product_, ctx_product_
                    )


def substitute_brand(brand, idx_brand, ctx_brand, idx_product, ctx_product):
    ctx_brand_cp = ctx_brand.copy()
    ctx_product_cp = ctx_product.copy()

    if ' '.join(ctx_brand_cp) == ' '.join(ctx_product_cp):
        brand_len = len(brand.split(' '))
        if brand_len > 1:
            if idx_brand < idx_product:
                ctx_brand_cp[idx_brand:idx_brand + 1] = brand.split(' ')
                ctx_brand_cp[idx_product+brand_len-1:idx_product+brand_len] = ctx_product_cp[idx_product].split(' ')
            return idx_brand, ctx_brand_cp, idx_product, ctx_brand_cp
        else:
            ctx_brand_cp[idx_brand:idx_brand + 1] = brand.split(' ')
            ctx_product_cp = ctx_brand_cp
            return idx_brand, ctx_brand_cp, idx_product, ctx_brand_cp
    else:
        ctx_brand_cp[idx_brand:idx_brand + 1] = brand.split(' ')
        ctx_product_cp[idx_product:idx_product + 1] = ctx_product_cp[idx_product].split(' ')

    return idx_brand, ctx_brand_cp, idx_product, ctx_product_cp



def main(argv=None):
    args = get_args(argv)
    brand_products_dict = get_brand_products_dict(args.source_file)
    substitute(args.source_file, brand_products_dict, args.sample_size)


if __name__ == "__main__":
    main()
