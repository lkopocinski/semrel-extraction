#!/usr/bin/env python3.6
from pathlib import Path
from typing import List, Iterator

import click
import torch

import data.scripts.utils.vectorizers as vec
from data.scripts.utils.corpus import DocSentence, Document, sentences_documents_gen, relations_file_paths
from data.scripts.utils.io import save_lines, save_tensor


class MapMaker:

    def __init__(self, vectorizer: vec.Vectorizer):
        self._vectorizer = vectorizer

    def _make_sentence_map(self, sentence: DocSentence, document: Document):
        keys = []
        vectors = []

        id_domain = document.directory
        id_document = document.id
        id_sentence = sentence.id

        for idx, orth in enumerate(sentence.orths):
            id_token = str(idx)
            key = (id_domain, id_document, id_sentence, id_token)
            vector = self._vectorizer.embed(sentence.orths)

            keys.append(key)
            vectors.append(vector)

        return keys, vectors

    def make_map(self, corpus_files: Iterator[Path]) -> [List[tuple], torch.Tensor]:
        keys = []
        vectors = []

        for sentence, document in sentences_documents_gen(corpus_files):
            _keys, _vectors = self._make_sentence_map(sentence, document)
            keys.extend(_keys)
            vectors.extend(_vectors)

        return keys, torch.cat(vectors)


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to corpora files.')
@click.option('--directories', required=True, nargs=3,
              help='Directories names with corpus files.')
@click.option('--elmo-model', required=True, type=(str, str),
              help="A path to elmo model options, weights.")
@click.option('--fasttext-model', required=True, type=str,
              help="A path to fasttext model.")
@click.option('--retrofit-model', required=True, type=str,
              help="File with retrofitted fasttext model.")
@click.option('--output-path', required=True, type=str,
              help='Directory for saving map files.')
def main(input_path, directories, elmo_model,
         fasttext_model, retrofit_model, output_path):
    elmo_options, elmo_weights = elmo_model
    makers_dict = {
        'elmo': MapMaker(vectorizer=vec.ElmoVectorizer(elmo_options, elmo_weights)),
        'fasttext': MapMaker(vectorizer=vec.FastTextVectorizer(fasttext_model)),
        'retrofit': MapMaker(vectorizer=vec.RetrofitVectorizer(retrofit_model, fasttext_model))
    }
    relations_files = relations_file_paths(input_path, directories)

    for name, mapmaker in makers_dict.items():
        keys, vectors = mapmaker.make_map(relations_files)

        save_lines(Path(f'{output_path}/{name}.map.keys'), keys)
        save_tensor(Path(f'{output_path}/{name}.map.pt'), vectors)


if __name__ == '__main__':
    main()
