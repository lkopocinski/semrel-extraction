import argparse
from pathlib import Path

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

from io import save_lines, save_tensor
from utils.corpus import documents_gen, get_document_ids, get_context


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpusfiles', required=True, help='File with corpus documents paths.')
    parser.add_argument('--weights', required=True, help="File with weights to elmo model.")
    parser.add_argument('--options', required=True, help="File with options to elmo model.")
    parser.add_argument('--output-path', required=True, help='Directory for saving map files.')
    return parser.parse_args(argv)


class ElmoVectorizer():

    def __init__(self, options, weights):
        self.model = Elmo(options, weights, 1, dropout=0)

    def embed(self, context):
        character_ids = batch_to_ids([context])
        embeddings = self.model(character_ids)
        tensor = embeddings['elmo_representations'][0]
        return tensor.squeeze()


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
    keys, vectors = make_map(Path(args.corpusfiles), elmo)

    save_lines(Path(f'{args.output_path}/elmo.map.keys'), keys)
    save_tensor(Path(f'{args.output_path}/elmo.map.pt'), vectors)


if __name__ == '__main__':
    main()
