#!/usr/bin/env python3.6
from pathlib import Path

import click

from data.scripts.relations import RelationsGenerator
from data.scripts.utils.corpus import from_index_documents_gen
from data.scripts.utils.io import save_lines, save_line

CHANNELS = (('BRAND_NAME', 'PRODUCT_NAME'),
            ('PRODUCT_NAME', 'BRAND_NAME'))

HEADER = '\t'.join([
    'label', 'id_domain', 'id_document',
    'id_sentence_from', 'lemma_from', 'channel_from', 'is_named_entity_from', 'indices_from', 'context_from',
    'id_sentence_to', 'lemma_to', 'channel_to', 'is_named_entity_to', 'indices_to', 'context_to'
])


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora relations files list.')
@click.option('--output-path', required=True, type=str,
              help='File path for saving generated relations files.')
def main(input_path, output_path):
    documents = from_index_documents_gen(relations_files_index=Path(input_path))
    output_path = Path(output_path)

    save_line(output_path, HEADER)

    for document in documents:
        relations_generator = RelationsGenerator(document=document)

        positive_relations = relations_generator.generate_positive(channels=CHANNELS)
        positives_lines = [f'in_relation\t{document.directory}\t{relation}'
                           for relation in positive_relations]

        negative_relations = relations_generator.generate_negative(channels=CHANNELS)
        negatives_lines = [f'no_relation\t{document.directory}\t{relation}'
                           for relation in negative_relations]

        save_lines(output_path, positives_lines + negatives_lines, mode='a')


if __name__ == "__main__":
    main()