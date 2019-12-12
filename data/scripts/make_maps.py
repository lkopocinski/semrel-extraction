#!/usr/bin/env python3

from pathlib import Path

import click
import torch

from io import save_lines, save_tensor
from utils.corpus import documents_gen, get_document_ids, get_context
from vectorizers import Vectorizer, ElmoVectorizer, FastTextVectorizer, RetrofitVectorizer


def get_key(document, sentence):
    id_domain, id_doc = get_document_ids(document)
    id_sent = sentence.id()
    context = get_context(sentence)
    return id_domain, id_doc, id_sent, context


def make_map(corpus_files: Path, vectorizer: Vectorizer):
    keys = []
    vectors = torch.FloatTensor()

    key_context = [
        get_key(document, sentence)
        for document in documents_gen(corpus_files)
        for paragraph in document.paragraphs()
        for sentence in paragraph.sentences()
    ]

    for id_domain, id_doc, id_sent, context in key_context:
        keys.extend([
            (id_domain, id_doc, id_sent, str(id_tok), orth)
            for id_tok, orth in enumerate(context)
        ])
        context_tensor = vectorizer.embed(context)
        torch.cat([vectors, context_tensor])

    return keys, vectors


@click.command()
@click.option('--corpusfiles', required=True, type=str, help='File with corpus documents paths.')
@click.option('--elmo_model', required=True, type=(str, str), help="A path to elmo model options, weight")
@click.option('--fasttext_model', required=True, type=str, help="A path to fasttext model")
@click.option('--retrofit-model', required=True, help="File with retrofitted fasttext model.")
@click.option('--output-path', required=True, help='Directory for saving map files.')
def main(corpusfiles, elmo_model, fasttext_model, retrofit_model, output_path):
    elmo = ElmoVectorizer(*elmo_model)
    fasttext = FastTextVectorizer(fasttext_model)
    retrofit = RetrofitVectorizer(retrofit_model, fasttext_model)

    for vectorizer, save_name in [(elmo, 'elmo'), (fasttext, 'fasttext'), (retrofit, 'retrofit')]:
        keys, vectors = make_map(Path(corpusfiles), vectorizer)

        save_lines(Path(f'{output_path}/{save_name}.map.keys'), keys)
        save_tensor(Path(f'{output_path}/{save_name}.map.pt'), vectors)


if __name__ == '__main__':
    main()
