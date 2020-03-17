#!/usr/bin/env python3.6

from pathlib import Path

import click

from . import constant
from .constant import RelationHeader as Rh
from .relations import RelationsGenerator
from .utils.corpus import from_index_documents_gen
from .utils.io import save_lines, save_line


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora relations files list.')
@click.option('--output-path', required=True, type=str,
              help='File path for saving generated relations files.')
def main(input_path, output_path):
    documents = from_index_documents_gen(relations_files_index=Path(input_path))
    output_path = Path(output_path)

    save_line(output_path, Rh.HEADER)

    for document in documents:
        relations_generator = RelationsGenerator(document=document)

        positive_relations = relations_generator.generate_positive(
            channels=constant.CHANNELS
        )
        positives_lines = [f'{constant.IN_RELATION_LABEL}'
                           f'\t{document.directory}'
                           f'\t{relation}'
                           for relation in positive_relations]

        negative_relations = relations_generator.generate_negative(
            channels=constant.CHANNELS
        )
        negatives_lines = [f'{constant.NO_RELATION_LABEL}'
                           f'\t{document.directory}'
                           f'\t{relation}'
                           for relation in negative_relations]

        save_lines(output_path, positives_lines + negatives_lines, mode='a')


if __name__ == "__main__":
    main()
