#!/usr/bin/env python3.6

import csv
from pathlib import Path
from typing import List

import click
import torch
import torch.nn as nn


from data.scripts.entities import Member, Relation
from data.scripts.utils.io import save_lines, save_tensor

PHRASE_LENGTH_LIMIT = 5


class RelationsLoader:

    def __init__(self, relations_path: Path):
        self._path_csv = relations_path

    def relations(self):
        with self._path_csv.open('r', newline='', encoding='utf-8') as file_csv:
            reader_csv = csv.DictReader(file_csv, delimiter='\t')
            for line_dict in reader_csv:
                relation = self._parse_relation(line_dict)
                yield line_dict['label'], line_dict['id_domain'], relation

    @staticmethod
    def _parse_relation(relation_dict: dict) -> Relation:
        member_from = Member(
            id_sentence=relation_dict['id_sent_1'],
            lemma=relation_dict['lemma_1'],
            channel=relation_dict['channel_1'],
            is_named_entity=relation_dict['is_named_entity_1'],
            indices=eval(relation_dict['indices_1']),
            context=eval(relation_dict['context_1'])
        )
        member_to = Member(
            id_sentence=relation_dict['id_sent_2'],
            lemma=relation_dict['lemma_2'],
            channel=relation_dict['channel_2'],
            is_named_entity=relation_dict['is_named_entity_2'],
            indices=eval(relation_dict['indices_2']),
            context=eval(relation_dict['context_2'])
        )
        return Relation(relation_dict['id_document'], member_from, member_to)


class MapLoader:

    def __init__(self, keys_file: str, vectors_file: str):
        self._keys_path = Path(keys_file)
        self._vectors_file = Path(vectors_file)

    def _load_keys(self) -> dict:
        with self._keys_path.open('r', encoding='utf-8') as file:
            return {
                eval(line.strip()): index
                for index, line in enumerate(file)
            }

    def _load_vectors(self) -> torch.Tensor:
        return torch.load(self._vectors_file)

    def __call__(self) -> [dict, torch.Tensor]:
        keys = self._load_keys()
        vectors = self._load_vectors()
        return keys, vectors


class RelationsVectorizer:

    def __init__(self, relations_loader: RelationsLoader, keys: dict, vectors: torch.Tensor):
        self.relations_loader = relations_loader

        self._keys = keys
        self._vectors = vectors

    @staticmethod
    def _max_pool(tensor: torch.Tensor):
        pool = nn.MaxPool1d(PHRASE_LENGTH_LIMIT, stride=0)
        tensor = tensor.transpose(2, 1)
        output = pool(tensor)
        return output.transpose(2, 1).squeeze()

    def _max_pool_vectors(self, tensor: List):
        vec1, vec2 = zip(*tensor)

        vec1 = torch.cat(vec1)
        vec2 = torch.cat(vec2)

        pooled1 = self._max_pool(vec1)
        pooled2 = self._max_pool(vec2)

        return torch.cat([pooled1, pooled2], dim=1)

    def _get_vectors_indices(self, id_domain, id_document, id_sentence, token_indices):
        return [self._keys[(id_domain, id_document, id_sentence, id_token)]
                for id_token in token_indices]

    def _get_tensor(self, vectors_indices):
        tensor = torch.zeros(1, PHRASE_LENGTH_LIMIT, self._vectors.shape[-1])
        vectors = self._vectors[vectors_indices]
        tensor[:, 0:self._vectors.shape[1], :] = vectors
        return tensor

    def member_to_key(self, member: Member):
        return member.id_sentence, member.channel, member.indices, member.lemma

    def make_tensors(self):
        keys = []
        vectors = []

        for label, id_domain, relation in self.relations_loader.relations():
            id_document, member_from, member_to = relation

            if len(member_from.indices) > PHRASE_LENGTH_LIMIT or len(member_to.indices) > PHRASE_LENGTH_LIMIT:
                continue

            keys.append((label, id_domain, relation.id_document,
                         self.member_to_key(member_from),
                         self.member_to_key(member_to)))

            vectors_indices_from = self._get_vectors_indices(id_domain, id_document,
                                                             member_from.id_sentence, member_from.indices)
            vectors_indices_to = self._get_vectors_indices(id_domain, id_document,
                                                           member_to.id_sentence, member_to.indices)

            vectors.append((self._get_tensor(vectors_indices_from),
                            self._get_tensor(vectors_indices_to)))

        max_pooled_tensor = self._max_pool_vectors(vectors)

        return keys, max_pooled_tensor


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
def main(
        input_path,
        elmo_map,
        fasttext_map,
        retrofit_map,
        output_path
):
    map_loaders = {
        'elmo': MapLoader(*elmo_map),
        'fasttext': MapLoader(*fasttext_map),
        'retrofit': MapLoader(*retrofit_map)
    }
    relations_path = Path(input_path)

    for name, load_map in map_loaders.items():
        keys, vectors = load_map()
        relations_loader = RelationsLoader(relations_path)
        vectorizer = RelationsVectorizer(relations_loader, keys, vectors)

        relations_keys, relations_vectors = vectorizer.make_tensors()

        save_lines(Path(f'{output_path}/{name}.rel.keys'), relations_keys)
        save_tensor(Path(f'{output_path}/{name}.rel.pt'), relations_vectors)


if __name__ == '__main__':
    main()
