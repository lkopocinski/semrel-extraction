#!/usr/bin/env python3.6
from itertools import chain
from pathlib import Path
from typing import List, Iterator

import click

from corpus import relations_documents_gen
from data.scripts.generator import RelationsGenerator
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

    for document in relations_documents_gen(relations_files):
        relations_generator = RelationsGenerator(document)

        positives = relations_generator.generate_positive()
        positives_lines = [f'in_relation\t{document.directory}\t{relation}' for relation in positives]

        negatives = relations_generator.generate_negative()
        negatives_lines = [f'no_relation\t{document.directory}\t{relation}' for relation in negatives]

        save_lines(out_path, positives_lines + negatives_lines, mode='a')


if __name__ == "__main__":
    main()
