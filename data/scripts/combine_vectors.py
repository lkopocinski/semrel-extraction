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

        with keys_file.open('r', newline='', encoding='utf-8') as file_csv:
            reader_csv = csv.reader(file_csv, delimiter='\t')
            return {key: idx for idx, key in enumerate(reader_csv)}


class MapLoader:

    def __init__(self, keys_file, vectors_file):
        self._keys_path = Path(keys_file)
        self._vectors_file = Path(vectors_file)

    def _load_keys(self) -> dict:
        with self._keys_path.open('r', encoding='utf-8') as file:
            return {eval(line.strip()): idx
                    for idx, line in enumerate(file)}

    def _load_vectors(self) -> torch.Tensor:
        return torch.load(self._vectors_file)

    def __call__(self) -> [dict, torch.Tensor]:
        keys = self._load_keys()
        vectors = self._load_vectors()
        return keys, vectors


class RelationsVectorizer:

    def __init__(self, keys: dict, vectors: torch.Tensor):
        self._keys = keys
        self._vectors = vectors

    @staticmethod
    def _max_pool(tensor: torch.Tensor):
        pool = nn.MaxPool1d(5, stride=0)
        tensor = tensor.transpose(2, 1)
        output = pool(tensor)
        return output.transpose(2, 1).squeeze()

    def make_tensors(self, relations_path: Path):
        relations_keys = []
        relations_vectors = []

        for label, id_domain, id_document, \
            id_sentence_1, lemma_1, channel_1, _, token_indices_1, _, \
            id_sentence_2, lemma_2, channel_2, _, token_indices_2, _ \
                in file_rows(relations_path):

            if len(eval(token_indices_1)) > 5 or len(eval(token_indices_2)) > 5:
                continue

            relations_keys.append((
                label, id_domain, id_document,
                id_sentence_1, lemma_1, channel_1, token_indices_1,
                id_sentence_2, lemma_2, channel_2, token_indices_2
            ))
            relations_vectors.append((
                self._get_tensor(id_domain, id_document, id_sentence_1, eval(token_indices_1)),
                self._get_tensor(id_domain, id_document, id_sentence_2, eval(token_indices_2))
            ))

        vec1, vec2 = zip(*relations_vectors)
        vec1, vec2 = torch.cat(vec1), torch.cat(vec2)
        pooled1, pooled2 = self._max_pool(vec1), self._max_pool(vec2)
        relations_vectors = torch.cat([pooled1, pooled2], dim=1)

        return relations_keys, relations_vectors

    def _get_tensor(self, id_domain, id_doc, id_sent, token_indices):
        vectors_indices = [
            self._keys[(id_domain, id_doc, id_sent, id_token)]
            for id_token in token_indices
        ]
        tensor = torch.zeros(1, 5, self._vectors.shape[-1])
        vectors = self._vectors[vectors_indices]
        tensor[:, 0:self._vectors.shape[1], :] = vectors
        return tensor


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--elmo-map', required=True, type=(str, str),
              metavar='elmo.map.pt elmo.map.keys',
              help="Elmo vectors and keys files.")
@click.option('--fasttext-map', required=True, type=(str, str),
              metavar='fasttext.map.pt fasttext.map.keys',
              help="Fasttext vectors and keys files.")
@click.option('--retrofit-map', required=True, type=(str, str),
              metavar='retrofit.map.pt retrofit.map.keys',
              help="Retrofit vectors and keys files.")
@click.option('--output-path', required=True, type=str,
              help='Directory for saving generated relations vectors.')
def main(input_path, output_path, elmo_map, fasttext_map, retrofit_map):
    maps_dict = {
        'elmo': MapLoader(*elmo_map),
        'fasttext': MapLoader(*fasttext_map),
        'retrofit': MapLoader(*retrofit_map)
    }
    relations_path = Path(input_path)

    for name, load_map in maps_dict.items():
        keys, vectors = load_map()
        vectorizer = RelationsVectorizer(keys, vectors)
        relations_keys, relations_vectors = vectorizer.make_tensors(relations_path)

        save_lines(Path(f'{output_path}/{name}.rel.keys'), relations_keys)
        save_tensor(Path(f'{output_path}/{name}.rel.pt'), relations_vectors)


if __name__ == '__main__':
    main()
