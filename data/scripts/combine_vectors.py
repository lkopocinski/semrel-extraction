#!/usr/bin/env python3
import csv
from pathlib import Path

import click
import torch
import torch.nn as nn

from io import save_lines, save_tensor


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


def make_relations_tensors(relations_path: Path, keys, vectors):
    rel_keys = []
    rel_vectors = []
    for id_domain, label, id_doc, id_sent1, lemma1, channel1, _, tokens1, _, id_sent2, lemma2, channel2, _, tokens2 in file_rows(
            relations_path):

        if len(eval(tokens1)) > 5 or len(eval(tokens2)) > 5:
            continue

        rel_keys.append((
            id_domain, label, id_doc, id_sent1, id_sent2, channel1, channel2, tokens1, tokens2, lemma1, lemma2
        ))
        rel_vectors.append((
            get_tensor(id_domain, id_doc, id_sent1, eval(tokens1), keys, vectors),
            get_tensor(id_domain, id_doc, id_sent2, eval(tokens2), keys, vectors)
        ))

    return rel_keys, rel_vectors


@click.command()
@click.option('--data-in', required=True, type=str,
              help='File with relations.')
@click.option('--output-path', required=True, type=str,
              help='Directory to save vectors.')
@click.option('--elmo-map', required=True, type=(str, str),
              metavar='elmo.map.pt elmo.map.keys',
              help="Elmo vectors and keys files.")
@click.option('--fasttext-map', required=True, type=(str, str),
              metavar='fasttext.map.pt fasttext.map.keys',
              help="Fasttext vectors and keys files.")
@click.option('--retrofit-map', required=True, type=(str, str),
              metavar='retrofit.map.pt retrofit.map.keys',
              help="Retrofit vectors and keys files.")
def main(data_in, output_path, elmo_map, fasttext_map, retrofit_map):
    maps_dict = {
        'elmo': load_map(*elmo_map),
        'fasttext': load_map(*fasttext_map),
        'retrofit': load_map(*retrofit_map)
    }
    relations_path = Path(data_in)

    for name, (keys, vectors) in maps_dict.items():
        keys, vectors = make_relations_tensors(relations_path, keys, vectors)
        vec1, vec2 = zip(*vectors)
        vec1, vec2 = torch.cat(vec1), torch.cat(vec2)
        pooled1, pooled2 = max_pool(vec1), max_pool(vec2)
        vectors = torch.cat([pooled1, pooled2], dim=1)

        save_lines(Path(f'{output_path}/{name}.rel.keys'), keys)
        save_tensor(Path(f'{output_path}/{name}.rel.pt'), vectors)


if __name__ == '__main__':
    main()
