#!/usr/bin/env python3
from pathlib import Path

import click

from generator import generate
from utils.io import save_lines


@click.command()
@click.option(
    '--data-in',
    required=True,
    type=str,
    help='Directory with corpora.'
)
@click.option(
    '--directories',
    required=True,
    nargs='+',
    help='Directories names with corpus files.'
)
@click.option(
    '--output-path',
    required=True,
    type=str,
    help='Directory for saving generated relations file.'
)
def main(data_in, directories, output_path):
    source_paths = [
        dir_path
        for dir_path in Path(data_in).iterdir()
        if dir_path.stem in directories
    ]
    out_path = Path(output_path)

    for path in source_paths:
        relations_files = list(path.glob('*.ne.rel.xml'))
        lines = generate(relations_files, ('BRAND_NAME', 'PRODUCT_NAME'))
        save_lines(out_path, lines, 'a')


if __name__ == "__main__":
    main()
