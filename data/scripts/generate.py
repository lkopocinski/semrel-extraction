#!/usr/bin/env python3
import argparse
from itertools import chain
from pathlib import Path

import argcomplete

from generator import generate_positive, generate_negative
from utils.io import save_lines


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with corpora.')
    parser.add_argument('--directories', nargs='+', required=True, help='Directories names with corpus files.')
    parser.add_argument('--output-path', required=True, help='Directory for saving generated relations file.')

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    lines = []
    out_file_path = Path(f'{args.output_path}/relations.tsv')
    for directory in args.directories:
        source_path = Path(f'{args.data_in}/{directory}')
        if source_path.is_dir():
            relations_files = list(source_path.glob('*.ne.rel.xml'))
            positive_lines = generate_positive(relations_files, ('BRAND_NAME', 'PRODUCT_NAME'))
            negative_lines = generate_negative(relations_files, ('BRAND_NAME', 'PRODUCT_NAME'))
            lines.extend([positive_lines, negative_lines])

    save_lines(out_file_path, chain(*lines))


if __name__ == "__main__":
    main()
