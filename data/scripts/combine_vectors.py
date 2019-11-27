import argparse
from pathlib import Path

import argcomplete


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='Directory with sampled datasets.')
    parser.add_argument('--output-path', required=True, help='Directory to save vectors.')
    parser.add_argument('--elmo-map', required=True, help="Map file with elmo vectors")
    parser.add_argument('--fasttext-map', required=True, help="Map file with fasttext vectors")
    parser.add_argument('--retrofit-map', required=True, help="Map file with retrofitted fasttext vectors")
    parser.add_argument('--sent2vec-map', required=True, help="Map file with sent2vec vectors")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def load_map(map_path: Path):
    vec_map = {}
    with map_path.open('r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split('\n')
            doc_id, sent_id, tok_id = row[0:3]
            vec = eval(row[4])
            vec_map[(doc_id, sent_id, tok_id)] = vec
    return vec_map


def combine_vectors(doc_id, sent_id, token_indices, vec_map):
    vectors = []
    for idx in token_indices:
        vec_map[(doc_id, sent_id, idx)]
        # TODO: MAX POOLING

    return []


def main(argv=None):
    args = get_args(argv)
    elmo_map = load_map(args.elmo_map)
    fasttext_map = load_map(args.fasttext_map)
    retrofit_map = load_map(args.retrofit_map)

    source_path = Path(f'{args.data_in}/relations.context')
    with source_path.open('r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split('\t')
            doc_id = row[2]
            sent_id = row[3]
            tokens_from = eval(row[7])
            tokens_to = eval(row[13])

            elmo_from_vec = combine_vectors(doc_id, sent_id, tokens_from, elmo_map)
            elmo_from_to = combine_vectors(doc_id, sent_id, tokens_to, elmo_map)

