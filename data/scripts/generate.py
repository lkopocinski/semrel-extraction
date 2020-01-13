#!/usr/bin/env python3.6
from itertools import chain
from pathlib import Path
from typing import List, Iterator

import click

from data.scripts.generator import generate
from data.scripts.utils.io import save_lines


def relations_file_paths(input_path: str, directories: List) -> Iterator[Path]:
    return chain.from_iterable(
        dir_path.glob('*.rel.xml')
        for dir_path in Path(input_path).iterdir()
        if dir_path.stem in directories)


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora files.')
@click.option('--directories', required=True, nargs=3,
              help='Directories names with corpus files.')
@click.option('--output-path', required=True, type=str,
              help='Directory for saving generated relations files.')
def main(input_path, directories, output_path):
    relations_files = relations_file_paths(input_path, directories)
    out_path = Path(output_path)

    lines = chain.from_iterable(generate(relations_files))
    save_lines(out_path, lines, mode='a')


if __name__ == "__main__":
    main()
