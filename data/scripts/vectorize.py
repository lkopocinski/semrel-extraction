#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import argcomplete
import numpy as np

import utils.vectorizers as v
from model.models import Relation
from utils.io import save_lines

np.set_printoptions(threshold=sys.maxsize)


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with sampled datasets.')
    parser.add_argument('--output-path', required=True, help='Directory to save vectors.')
    parser.add_argument('--elmo_weights', required=True, help="File with weights to elmo model.")
    parser.add_argument('--elmo_options', required=True, help="File with options to elmo model.")
    parser.add_argument('--fasttext_model', required=True, help="File with fasttext bin model.")
    parser.add_argument('--sent2vec_model', required=True, help="File with sent2vec model.")
    parser.add_argument('--retrofit_model', required=True, help="File with retrofitted model.")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    engines = {
        'elmo': v.ElmoVectorizer(args.elmo_options, args.elmo_weights),
        'elmoconv': v.ElmoConvolutionVectorizer(args.elmo_options, args.elmo_weights),
        'fasttext': v.FastTextVectorizer(args.fasttext_model),
        'sent2vec': v.Sent2VecVectorizer(args.sent2vec_model),
        'retrofit': v.RetrofitVectorizer(args.retrofit_model, args.fasttext_model),
        'ner': v.NamedEntityVectorizer()
    }

    for set_name in ['train', 'valid', 'test']:
        for label_type, label_name in [('positive', 'in_relation'), ('negative', 'no_relation')]:
            source_path = Path(f'{args.data_in}/{set_name}/{label_type}')
            if source_path.is_dir():
                for file_path in source_path.glob('*.sampled'):
                    out_file_path = Path(f'{args.output_path}/{set_name}/{label_type}/{file_path.stem}.vectors')
                    lines = create_vectors(engines, file_path, label_name)
                    save_lines(out_file_path, lines)


def create_vectors(engines: dict, path: Path, relation_type: str):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            relation = Relation.from_line(line)

            elmo1, elmo2 = engines['elmo'].make_vectors(relation)
            elmoconv1, elmoconv2 = engines['elmoconv'].make_vectors(relation)
            fasttext1, fasttext2 = engines['fasttext'].make_vectors(relation)
            sent2vec1, sent2vec2 = engines['sent2vec'].make_vectors(relation)
            retrofit1, retrofit2 = engines['retrofit'].make_vectors(relation)
            ner1, ner2 = engines['ner'].make_vectors(relation)

            yield f'{relation_type}\t' \
                  f'{relation}\t' \
                  f'{elmo1}\t{elmo2}\t' \
                  f'{elmoconv1}\t{elmoconv2}\t' \
                  f'{fasttext1}\t{fasttext2}\t' \
                  f'{sent2vec1}\t{sent2vec2}\t' \
                  f'{retrofit1}\t{retrofit2}\t' \
                  f'{ner1}\t{ner2}\t'


if __name__ == "__main__":
    main()
