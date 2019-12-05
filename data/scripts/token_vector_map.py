import argparse
from pathlib import Path

import torch
from gensim.models.fasttext import load_facebook_model

from io import save_lines, save_tensor
from utils.corpus import documents_gen, get_document_ids, get_context


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpusfiles', required=True, help='File with corpus documents paths.')
    parser.add_argument('--model', required=True, help="File with fasttext model.")
    parser.add_argument('--output-path', required=True, help='Directory for saving map files.')
    return parser.parse_args(argv)


class FastTextVectorizer():

    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)

    def embed(self, context):
        return torch.FloatTensor(self.model.wv[context])


def get_key(document, sentence):
    id_domain, id_doc = get_document_ids(document)
    id_sent = sentence.id()
    context = get_context(sentence)
    return id_domain, id_doc, id_sent, context


def make_map(corpus_files: Path, vectorizer: FastTextVectorizer):
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
    elmo = FastTextVectorizer(args.model)
    keys, vectors = make_map(Path(args.corpusfiles), elmo)

    save_lines(Path(f'{args.output_path}/fasttext.map.keys'), keys)
    save_tensor(Path(f'{args.output_path}/fasttext.map.pt'), vectors)


if __name__ == '__main__':
    main()
