#!/usr/bin/env python3

import argparse
from pathlib import Path

import argcomplete
import torch.nn as nn
import torch
from model.models import Vector


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with sampled datasets.')
    parser.add_argument('--output-path', required=True, help='Directory to save vectors.')
    parser.add_argument('--elmo-map', required=True, help="Map file with elmo vectors")
    # parser.add_argument('--elmoconv-map', required=True, help="Map file with elmoconv vectors")
    parser.add_argument('--fasttext-map', required=True, help="Map file with fasttext vectors")
    parser.add_argument('--retrofit-map', required=True, help="Map file with retrofitted fasttext vectors")
    # parser.add_argument('--sent2vec-map', required=True, help="Map file with sent2vec vectors")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def file_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            yield line.strip().split('\t')


def load_map(map_path):
    vec_map = {}
    for row in file_rows(Path(map_path)):
        try:
            doc_id, sent_id, tok_id = row[0:3]
            vec = eval(row[4])
            vec_map[(doc_id, sent_id, int(tok_id))] = vec
        except ValueError:
            continue
    return vec_map


def combine_vectors(doc_id, sent_id, token_indices, vec_map):
    pool = nn.MaxPool1d(3, stride=2)
    vectors = [vec_map[(doc_id, sent_id, idx)] for idx in token_indices]
    vec = torch.FloatTensor(vectors).unsqueeze(0)
    output = pool(vec)
    return str(Vector(output.numpy()))


def main(argv=None):
    args = get_args(argv)
    import pudb
    pudb.set_trace()
    elmo_map = load_map(args.elmo_map)
    # elmoconv_map = load_map(args.elmoconv_map)
    fasttext_map = load_map(args.fasttext_map)
    retrofit_map = load_map(args.retrofit_map)
    # sent2vec_map = load_map(args.sent2vec_map)

    source_path = Path(f'{args.data_in}/relations.fake.context')
    for row in file_rows(source_path):
        doc_id = row[2]
        sent_id = row[3]
        tokens_from = eval(row[7])
        tokens_to = eval(row[13])

        elmo_from_vec = combine_vectors(doc_id, sent_id, tokens_from, elmo_map)
        elmo_to_vec = combine_vectors(doc_id, sent_id, tokens_to, elmo_map)

        # elmoconv_from_vec = combine_vectors(doc_id, sent_id, tokens_from, elmoconv_map)
        # elmoconv_to_vec = combine_vectors(doc_id, sent_id, tokens_to, elmoconv_map)

        fasttext_from_vec = combine_vectors(doc_id, sent_id, tokens_from, fasttext_map)
        fasttext_to_vec = combine_vectors(doc_id, sent_id, tokens_from, fasttext_map)

        retrofit_from_vec = combine_vectors(doc_id, sent_id, tokens_from, retrofit_map)
        retrofit_to_vec = combine_vectors(doc_id, sent_id, tokens_from, retrofit_map)

        # sent2vec_from_vec = combine_vectors(doc_id, sent_id, tokens_from, sent2vec_map)
        # sent2vec_to_vec = combine_vectors(doc_id, sent_id, tokens_from, sent2vec_map)

        ne_from = str([float(eval(row[6]))])
        ne_to = str([float(eval(row[12]))])

        row.extend([
            elmo_from_vec, elmo_to_vec,
            "[0, 0, 0]", "[0, 0, 0]",
            fasttext_from_vec, fasttext_to_vec,
            retrofit_from_vec, retrofit_to_vec,
            "[0, 0, 0]", "[0, 0, 0]",
            ne_from, ne_to
        ])

        print('\t'.join(row))


if __name__ == '__main__':
    main()
