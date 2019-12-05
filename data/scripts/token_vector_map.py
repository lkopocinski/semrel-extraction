import argparse
from pathlib import Path

import torch
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model

from io import save_lines, save_tensor
from utils.corpus import documents_gen, get_document_ids, get_context


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpusfiles', required=True, help='File with corpus documents paths.')
    parser.add_argument('--model-retrofitted', required=True, help="File with retrofitted fasttext model.")
    parser.add_argument('--model-fasttext', required=True, help="File with fasttext model.")
    parser.add_argument('--output-path', required=True, help='Directory for saving map files.')
    return parser.parse_args(argv)


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


def make_map(corpus_files: Path, vectorizer: RetrofitVectorizer):
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
    elmo = RetrofitVectorizer(args.model_retrofitted, args.fasttext_model)
    keys, vectors = make_map(Path(args.corpusfiles), elmo)

    save_lines(Path(f'{args.output_path}/retrofit.map.keys'), keys)
    save_tensor(Path(f'{args.output_path}/retrofit.map.pt'), vectors)


if __name__ == '__main__':
    main()
