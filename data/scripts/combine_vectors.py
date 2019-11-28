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


def combine_vectors(tensor):
    pool = nn.MaxPool1d(5, stride=0)
    tensor = tensor.transpose(2, 1)
    output = pool(tensor)
    return output


def get_tensor(doc_id, sent_id, token_indices, vec_map):
    vectors = [vec_map[(doc_id, sent_id, idx)] for idx in token_indices]
    vectors = torch.FloatTensor(vectors).unsqueeze(0)
    tensor = torch.zeros(1, 5, 1024)
    tensor[:, 0:vectors.shape[1], :] = vectors
    return tensor


def main(argv=None):
    args = get_args(argv)
    import pudb
    pudb.set_trace()
    elmo_map = load_map(args.elmo_map)
    # elmoconv_map = load_map(args.elmoconv_map)
    # fasttext_map = load_map(args.fasttext_map)
    # retrofit_map = load_map(args.retrofit_map)
    # sent2vec_map = load_map(args.sent2vec_map)

    source_path = Path(f'{args.data_in}/relations.fake.context')

    rel_map = {}
    for row in file_rows(source_path):
        doc_id = row[2]

        sent_id1 = row[3]
        tokens1 = eval(row[7])

        sent_id2 = row[9]
        tokens2 = eval(row[13])

        rel_key = (doc_id, (sent_id1, tuple(tokens1)), (sent_id2,
                                                        tuple(tokens2)))
        rel_map[rel_key] = (get_tensor(doc_id, sent_id1, tokens1, elmo_map),
                            get_tensor(doc_id, sent_id2, tokens2, elmo_map))

        elmo1, elmo2 = zip(*rel_map.values())
        output = combine_vectors(torch.cat(elmo1))
        print(output)


if __name__ == '__main__':
    main()
