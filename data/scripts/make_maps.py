#!/usr/bin/env python3

from pathlib import Path

import click
import torch

import data.scripts.utils.corpus as corp
import data.scripts.utils.vectorizers as vec
from data.scripts.utils.io import save_lines, save_tensor


class MapMaker:

    def __init__(self, vectorizer: vec.Vectorizer):
        self._vectorizer = vectorizer

    def _make_sentence_map(self, sentence, document):
        keys = []
        vectors = []

        context = corp.get_context(sentence)
        for idx, orth in enumerate(context):
            id_domain = corp.get_document_dir(document)
            id_document = corp.get_document_file_name(document)
            id_sentence = corp.get_sentence_id(sentence)
            id_token = str(idx)

            key = (id_domain, id_document, id_sentence, id_token, orth)
            vector = self._vectorizer.embed(context)

            keys.append(key)
            vectors.append(vector)

        return keys, vectors

    def make_map(self, corpus_files: Path):
        sentences_documents = (
            (sentence, document)
            for document in corp.documents_gen(corpus_files)
            for paragraph in document.paragraphs()
            for sentence in paragraph.sentences()
        )

        keys = []
        vectors = []

        for sentence, document in sentences_documents:
            _keys, _vectors = self._make_sentence_map(self, sentence, document)
            keys.extend(_keys)
            vectors.extend(_vectors)

        return keys, torch.cat(vectors)


@click.command()
@click.option(
    '--corpus-files',
    required=True,
    type=str,
    help='File with corpus documents paths.'
)
@click.option(
    '--elmo_model',
    required=True,
    type=(str, str),
    help="A path to elmo model options, weight"
)
@click.option(
    '--fasttext_model',
    required=True,
    type=str,
    help="A path to fasttext model"
)
@click.option(
    '--retrofit-model',
    required=True,
    help="File with retrofitted fasttext model."
)
@click.option(
    '--output-path',
    required=True,
    help='Directory for saving map files.'
)
def main(corpus_files, elmo_model, fasttext_model,
         retrofit_model, output_path):
    elmo = MapMaker(
        vectorizer=vec.ElmoVectorizer(*elmo_model)
    )
    # fasttext = MapMaker(
    #     vectorizer=vec.FastTextVectorizer(fasttext_model)
    # )
    # retrofit = MapMaker(
    #     vectorizer=vec.RetrofitVectorizer(retrofit_model, fasttext_model)
    # )
    # corpus_files = Path(corpus_files)

    for mapmaker, save_name in [
        (elmo, 'elmo') #, (fasttext, 'fasttext'), (retrofit, 'retrofit')
    ]:
        keys, vectors = mapmaker.make_map(corpus_files)

        save_lines(Path(f'{output_path}/{save_name}.map.keys'), keys)
        save_tensor(Path(f'{output_path}/{save_name}.map.pt'), vectors)


if __name__ == '__main__':
    main()
