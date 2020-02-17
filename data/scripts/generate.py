#!/usr/bin/env python3.6
from pathlib import Path

import click

from data.scripts.generator import RelationsGenerator
from data.scripts.utils.corpus import relations_documents_from_index
from data.scripts.utils.io import save_lines

CHANNELS = (('BRAND_NAME', 'PRODUCT_NAME'),
            ('PRODUCT_NAME', 'BRAND_NAME'))


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora relations files list.')
@click.option('--output-path', required=True, type=str,
              help='Directory for saving generated relations files.')
def main(input_path, output_path):
    documents = relations_documents_from_index(index_path=Path(input_path))
    output_path = Path(output_path)

    for document in documents:
        relations_generator = RelationsGenerator(document)

        positive_relations = relations_generator.generate_positive(CHANNELS)
        positives_lines = [f'in_relation\t{document.directory}\t{relation}' for relation in positive_relations]

        negative_relations = relations_generator.generate_negative(CHANNELS)
        negatives_lines = [f'no_relation\t{document.directory}\t{relation}' for relation in negative_relations]

        save_lines(output_path, positives_lines + negatives_lines, mode='a')


if __name__ == "__main__":
    main()
