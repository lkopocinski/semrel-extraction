#!/usr/bin/env python3.6

from pathlib import Path
from typing import List

import click
import torch
import torch.nn as nn

from data.scripts.entities import Relation, Member
from data.scripts.maps import MapLoader
from data.scripts.relations import RelationsLoader
from data.scripts.utils.io import save_lines, save_tensor


class RelationsEmbedder:
    PHRASE_LENGTH_LIMIT = 5

    def __init__(self, relations_loader: RelationsLoader, keys: dict, vectors: torch.Tensor):
        self.relations_loader = relations_loader

        self._keys = keys
        self._vectors = vectors

    def _max_pool(self, tensor: torch.Tensor) -> torch.Tensor:
        pool = nn.MaxPool1d(self.PHRASE_LENGTH_LIMIT, stride=0)
        tensor = tensor.transpose(2, 1)
        output = pool(tensor)
        return output.transpose(2, 1).squeeze()

    def _max_pool_vectors(self, tensor: List) -> torch.Tensor:
        vector1, vector2 = zip(*tensor)

        vector1 = torch.cat(vector1)
        vector2 = torch.cat(vector2)

        pooled1 = self._max_pool(vector1)
        pooled2 = self._max_pool(vector2)

        return torch.cat([pooled1, pooled2], dim=1)

    def _get_vectors_indices(self, id_domain, id_document, member: Member) -> List[int]:
        return [self._keys[(id_domain, id_document, member.id_sentence, str(id_token))]
                for id_token in member.indices]

    def _get_tensor(self, vectors_indices: List[int]) -> torch.Tensor:
        tensor = torch.zeros(1, self.PHRASE_LENGTH_LIMIT, self._vectors.shape[-1])
        vectors = self._vectors[vectors_indices]
        tensor[:, 0:vectors.shape[0], :] = vectors
        return tensor

    @staticmethod
    def _make_relation_key(label: str, id_domain: str, relation: Relation) -> str:
        id_document, member_from, member_to = relation
        return '\t'.join([
            label, id_domain, id_document,
            member_from.id_sentence, member_from.channel, str(member_from.indices), member_from.lemma,
            member_to.id_sentence, member_to.channel, str(member_to.indices), member_to.lemma,
        ])

    def _is_phrase_too_long(self, member: Member) -> bool:
        return len(member.indices) > self.PHRASE_LENGTH_LIMIT

    def make_tensor(self) -> [List, torch.Tensor]:
        keys = []
        vectors = []

        for label, id_domain, relation in self.relations_loader.relations():
            id_document, member_from, member_to = relation

            if self._is_phrase_too_long(member_from) or self._is_phrase_too_long(member_to):
                continue

            vectors_indices_from = self._get_vectors_indices(id_domain, id_document, member_from)
            vectors_indices_to = self._get_vectors_indices(id_domain, id_document, member_to)

            vectors_from = self._get_tensor(vectors_indices_from)
            vectors_to = self._get_tensor(vectors_indices_to)

            key = self._make_relation_key(label, id_domain, relation)

            keys.append(key)
            vectors.append((vectors_from, vectors_to))

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
              help='Directory for saving relations embeddings.')
def main(input_path, elmo_map, fasttext_map, retrofit_map, output_dir):
    map_loaders = {
        'elmo': MapLoader(*elmo_map),
        'fasttext': MapLoader(*fasttext_map),
        'retrofit': MapLoader(*retrofit_map)
    }
    relations_path = Path(input_path)

    for name, load_map in map_loaders.items():
        keys, vectors = load_map()
        relations_loader = RelationsLoader(relations_path)
        relations_embedder = RelationsEmbedder(relations_loader, keys, vectors)

        relations_keys, relations_vectors = relations_embedder.make_tensor()

        save_lines(Path(f'{output_dir}/{name}.rel.keys'), relations_keys)
        save_tensor(Path(f'{output_dir}/{name}.rel.pt'), relations_vectors)


if __name__ == '__main__':
    main()
