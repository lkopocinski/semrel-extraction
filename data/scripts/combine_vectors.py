#!/usr/bin/env python3.6

import csv
from pathlib import Path
from typing import List

import click
import torch
import torch.nn as nn

from data.scripts.entities import Member, Relation
from data.scripts.maps import MapLoader
from data.scripts.utils.io import save_lines, save_tensor

PHRASE_LENGTH_LIMIT = 5


class RelationsLoader:

    def __init__(self, relations_path: Path):
        self._path_csv = relations_path

    def relations(self):
        with self._path_csv.open('r', newline='', encoding='utf-8') as file_csv:
            reader_csv = csv.DictReader(file_csv, delimiter='\t')
            for line_dict in reader_csv:
                label = line_dict['label']
                id_domain = line_dict['id_domain']
                relation = self._parse_relation(line_dict)

                yield label, id_domain, relation

    @staticmethod
    def _parse_relation(relation_dict: dict) -> Relation:
        id_document = relation_dict['id_document']
        member_from = Member(
            id_sentence=relation_dict['id_sentence_from'],
            lemma=relation_dict['lemma_from'],
            channel=relation_dict['channel_from'],
            is_named_entity=relation_dict['is_named_entity_from'],
            indices=eval(relation_dict['indices_from']),
            context=eval(relation_dict['context_from'])
        )
        member_to = Member(
            id_sentence=relation_dict['id_sentence_to'],
            lemma=relation_dict['lemma_to'],
            channel=relation_dict['channel_to'],
            is_named_entity=relation_dict['is_named_entity_to'],
            indices=eval(relation_dict['indices_to']),
            context=eval(relation_dict['context_to'])
        )

        return Relation(id_document, member_from, member_to)


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
        return [self._keys[(id_domain, id_document, id_sentence, str(id_token))]
                for id_token in token_indices]

    def _get_tensor(self, vectors_indices):
        tensor = torch.zeros(1, PHRASE_LENGTH_LIMIT, self._vectors.shape[-1])
        vectors = self._vectors[vectors_indices]
        tensor[:, 0:vectors.shape[0], :] = vectors
        return tensor

    def _make_key(self, label: str, id_domain: str, relation: Relation):
        id_document, member_from, member_to = relation
        return '\t'.join([
            label, id_domain, id_document,
            member_from.id_sentence, member_from.channel, str(member_from.indices), member_from.lemma,
            member_to.id_sentence, member_to.channel, str(member_to.indices), member_to.lemma,
        ])

    def make_tensors(self):
        keys = []
        vectors = []

        for label, id_domain, relation in self.relations_loader.relations():
            id_document, member_from, member_to = relation

            if len(member_from.indices) > PHRASE_LENGTH_LIMIT or len(member_to.indices) > PHRASE_LENGTH_LIMIT:
                continue

            keys.append(self._make_key(label, id_domain, relation))

            vectors_indices_from = self._get_vectors_indices(
                id_domain, id_document, member_from.id_sentence, member_from.indices)
            vectors_indices_to = self._get_vectors_indices(
                id_domain, id_document, member_to.id_sentence, member_to.indices)

            vectors.append((self._get_tensor(vectors_indices_from),
                            self._get_tensor(vectors_indices_to)))

        max_pooled_tensor = self._max_pool_vectors(vectors)

        return keys, max_pooled_tensor


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--elmo-map', required=True, type=(str, str),
              metavar='elmo.map.keys elmo.map.pt',
              help="Elmo keys and vectors files.")
@click.option('--fasttext-map', required=True, type=(str, str),
              metavar='fasttext.map.keys fasttext.map.pt',
              help="Fasttext keys and vectors files.")
@click.option('--retrofit-map', required=True, type=(str, str),
              metavar='retrofit.map.keys retrofit.map.pt',
              help="Retrofit keys and vectors files.")
@click.option('--output-dir', required=True, type=str,
              help='Directory for saving generated relations vectors.')
def main(
        input_path,
        elmo_map,
        fasttext_map,
        retrofit_map,
        output_dir
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

        save_lines(Path(f'{output_dir}/{name}.rel.keys'), relations_keys)
        save_tensor(Path(f'{output_dir}/{name}.rel.pt'), relations_vectors)


if __name__ == '__main__':
    main()
