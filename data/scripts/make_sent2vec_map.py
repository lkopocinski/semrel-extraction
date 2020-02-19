#!/usr/bin/env python3.6
from pathlib import Path

import click

import sent2vec
from data.scripts.combine_vectors import RelationsLoader
from data.scripts.utils.corpus import relations_documents_from_index


def make_sentence_map(relations_paths: Path):
    sentence_map = {}

    documents = relations_documents_from_index(index_path=relations_paths)
    for document in documents:
        id_domain = document.directory
        id_document = document.id
        sentence_map[(id_domain, id_document)] = {}

        for sentence in document.sentences:
            sentence_index = int(sentence.id.replace('sent', ''))
            context = sentence.orths
            sentence_map[(id_domain, id_document)][sentence_index] = context

    return sentence_map


@click.command()
@click.option('--relations-file', required=True, type=str,
              help='Path to relations file.')
@click.option('--documents-files', required=True, type=str,
              help='Path to relations file.')
@click.option('--model', required=True, type=str,
              metavar='model.bin',
              help="Paths to sent2vec model.")
@click.option('--output-paths', required=True, type=(str, str),
              metavar='sent2vec.map.keys sent2vec.map.pt',
              help='Paths for saving keys and map files.')
def main(relations_file, documents_files, model, output_paths):
    s2v = sent2vec.Sent2vecModel()
    s2v.load_model(model, inference_mode=True)

    sentence_map = make_sentence_map(documents_files)
    import pudb;
    pudb.set_trace()
    relations_loader = RelationsLoader(relations_file)


if __name__ == '__main__':
    main()
