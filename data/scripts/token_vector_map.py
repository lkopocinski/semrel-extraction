#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model

from io import save_lines, save_tensor
from utils.corpus import documents_gen, get_document_ids, get_context


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpusfiles', required=True, help='File with corpus documents paths.')
    parser.add_argument('--elmo-weights', required=True, help="File with weights to elmo model.")
    parser.add_argument('--elmo-options', required=True, help="File with options to elmo model.")
    parser.add_argument('--fasttext-model', required=True, help="File with fasttext model.")
    parser.add_argument('--retrofit-model', required=True, help="File with retrofitted fasttext model.")
    parser.add_argument('--output-path', required=True, help='Directory for saving map files.')
    return parser.parse_args(argv)


class ElmoVectorizer:

    def __init__(self, options, weights):
        self.model = Elmo(options, weights, 1, dropout=0)

    def embed(self, context):
        character_ids = batch_to_ids([context])
        embeddings = self.model(character_ids)
        tensor = embeddings['elmo_representations'][0]
        return tensor.squeeze()


class FastTextVectorizer:

    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)

    def embed(self, context):
        return torch.FloatTensor(self.model.wv[context])


class RetrofitVectorizer:

    def __init__(self, retrofitted_model_path, fasttext_model_path):
        self.model_retrofit = KeyedVectors.load_word2vec_format(retrofitted_model_path)
        self.model_fasttext = load_facebook_model(fasttext_model_path)

    def _embed_word(self, word):
        try:
            return torch.FloatTensor(self.model_retrofit[word])
        except KeyError:
            print("Term not found in retrofit model: ", word)
            return torch.FloatTensor(self.model_fasttext.wv[word])

    def embed(self, context):
        tensors = [self._embed_word(word) for word in context]
        return torch.stack(tensors)


def get_key(document, sentence):
    id_domain, id_doc = get_document_ids(document)
    id_sent = sentence.id()
    context = get_context(sentence)
    return id_domain, id_doc, id_sent, context


def make_map(corpus_files: Path, vectorizer: ElmoVectorizer):
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


def main(argv=None):
    args = get_args(argv)
    elmo = ElmoVectorizer(args.weights, args.options)
    fasttext = FastTextVectorizer(args.fasttext_model)
    retrofit = RetrofitVectorizer(args.retrofit_model, args.fasttext_model)

    for vectorizer, save_name in [(elmo, 'elmo'), (fasttext, 'fasttext'), (retrofit, 'retrofit')]:
        keys, vectors = make_map(Path(args.corpusfiles), vectorizer)

        save_lines(Path(f'{args.output_path}/{save_name}.map.keys'), keys)
        save_tensor(Path(f'{args.output_path}/{save_name}.map.pt'), vectors)


if __name__ == '__main__':
    main()
