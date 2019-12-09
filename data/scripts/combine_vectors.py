#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import argcomplete
import torch
import torch.nn as nn

from io import save_lines, save_tensor


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-in', required=True, help='File with relations.')
    parser.add_argument('--output-path', required=True, help='Directory to save vectors.')
    parser.add_argument('--elmo-map', required=True, nargs=2, metavar='elmo.map.pt elmo.map.keys',
                        help="Elmo vectors and keys files.")
    parser.add_argument('--fasttext-map', required=True, nargs=2, metavar='fasttext.map.pt fasttext.map.keys',
                        help="Fasttext vectors and keys files.")
    parser.add_argument('--retrofit-map', required=True, nargs=2, metavar='retrofit.map.pt retrofit.map.keys',
                        help="Retrofit vectors and keys files.")

    argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def file_rows(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            yield line.strip().split('\t')


def load_keys(keys_file):
    with open(keys_file, 'r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        return {key: idx for idx, key in enumerate(csvreader)}


def load_map(vectors_file, keys_file):
    keys = load_keys(keys_file)
    vectors = torch.load(vectors_file)
    return keys, vectors


def max_pool(tensor):
    pool = nn.MaxPool1d(5, stride=0)
    tensor = tensor.transpose(2, 1)
    output = pool(tensor)
    return output.transpose(2, 1).squeeze()


def get_tensor(id_domain, id_doc, id_sent, token_indices, keys, vectors_map):
    vectors_indices = [keys[(id_domain, id_doc, id_sent, idx)] for idx in token_indices]
    tensor = torch.zeros(1, 5, vectors_map.shape[-1])
    vectors = vectors_map[vectors_indices]
    tensor[:, 0:vectors_map.shape[1], :] = vectors
    return tensor


def make_relations_tensors(path: Path, keys, vectors):
    rel_keys = []
    rel_vectors = []
    for id_domain, label, id_doc, id_sent1, lemma1, channel1, _, tokens1, _, id_sent2, lemma2, channel2, _, tokens2 in file_rows(
            path):

        if len(eval(tokens1)) > 5 or len(eval(tokens2)) > 5:
            continue

        rel_keys.append((
            id_domain, label, id_doc, id_sent1, id_sent2, channel1, channel2, tokens1, tokens2, lemma1, lemma2
        ))
        rel_vectors.append((
            get_tensor(id_domain, id_doc, id_sent1, tokens1, keys, vectors),
            get_tensor(id_domain, id_doc, id_sent2, tokens2, keys, vectors)
        ))

    return rel_keys, rel_vectors


def main(argv=None):
    args = get_args(argv)
    keys_elmo, vectors_elmo = load_map(*args.elmo_map)
    keys_fasttext, vectors_fasttext = load_map(*args.fasttext_map)
    keys_retrofit, vectors_retrofit = load_map(*args.retrofit_map)

    relations_path = Path(args.data_in)

    keys, vectors = make_relations_tensors(relations_path, keys_elmo, vectors_elmo)
    vec1, vec2 = zip(*vectors)
    vec1, vec2 = torch.cat(vec1), torch.cat(vec2)
    pooled1, pooled2 = max_pool(vec1), max_pool(vec2)

    concat_dump = torch.cat([pooled1, pooled2], dim=1)

    save_lines(Path(f'{args.output_path}/elmo.rel.keys'), keys)
    save_tensor(Path(f'{args.output_path}/elmo.rel.pt'), concat_dump)


if __name__ == '__main__':
    main()
