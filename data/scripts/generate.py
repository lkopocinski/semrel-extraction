#!/usr/bin/env python3.6
from pathlib import Path

import click

from data.scripts.utils.corpus import relations_documents_gen, relations_file_paths
from data.scripts.generator import RelationsGenerator
from data.scripts.utils.io import save_lines

CHANNELS = (('BRAND_NAME', 'PRODUCT_NAME'),
            ('PRODUCT_NAME', 'BRAND_NAME'))


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora files.')
@click.option('--directories', required=True, nargs=1,
              help='Directories names with corpus files.')
@click.option('--output-path', required=True, type=str,
              help='Directory for saving generated relations files.')
def main(
        input_path,
        directories,
        output_path
):
    relations_files = relations_file_paths(input_path, directories)
    output_path = Path(output_path)

    for document in relations_documents_gen(relations_files):
        relations_generator = RelationsGenerator(document)

        positive_relations = relations_generator.generate_positive(CHANNELS)
        positives_lines = [f'in_relation\t{document.directory}\t{relation}' for relation in positive_relations]

        # negative_relations = relations_generator.generate_negative(CHANNELS)
        # negatives_lines = [f'no_relation\t{document.directory}\t{relation}' for relation in negative_relations]

        # save_lines(output_path, positives_lines + negatives_lines, mode='a')
        save_lines(output_path, positives_lines, mode='a')


if __name__ == "__main__":
    main()
