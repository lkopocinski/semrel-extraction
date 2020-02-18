#!/usr/bin/env python3.6
from pathlib import Path
from typing import Iterator

import click

import sent2vec
from data.scripts.utils.corpus import relations_documents_from_index, Document

from data.scripts.combine_vectors import RelationsLoader


def make_sentence_map(relations_loader):
    sentence_map = {}

    for label, id_domain, relation in relations_loader.relations():
        id_document, member_from, member_to = relation
        if (id_domain, id_document) not in sentence_map:
            document = cclutils.read_ccl(f'{corpus_path}/{domain}/{doc_id}.xml')
            sentence_map[(domain, doc_id)] = {
                int(sentence.id().replace('sent', '')): [token.orth_utf8() for token in sentence.tokens()]
                for par in document.paragraphs()
                for sentence in par.sentences()
            }



@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--model', required=True, type=(str, str),
              metavar='model.bin',
              help="Paths to sent2vec model.")
@click.option('--output-paths', required=True, type=(str, str),
              metavar='sent2vec.map.keys sent2vec.map.pt',
              help='Paths for saving keys and map files.')
def main(input_path, model, output_paths):
    model = sent2vec.Sent2vecModel()
    model.load_model(model, inference_mode=True)

    relations_loader = RelationsLoader(input_path)
    make_sentence_map(relations_loader)

    rel_map =


if __name__ == '__main__':
    main()
