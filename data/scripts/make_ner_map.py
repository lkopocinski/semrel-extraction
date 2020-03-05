#!/usr/bin/env python3.6
from pathlib import Path
from typing import List

import click
import torch

from data.scripts.entities import Relation, Member
from data.scripts.relations import RelationsLoader
from data.scripts.utils.io import save_lines, save_tensor


class RelationsMapMaker:
    PHRASE_LENGTH_LIMIT = 5

    def __init__(self, relations_loader: RelationsLoader):
        self.relations_loader = relations_loader

    def _is_phrase_too_long(self, member: Member) -> bool:
        return len(member.indices) > self.PHRASE_LENGTH_LIMIT

    @staticmethod
    def _make_key(label: str, id_domain: str, relation: Relation):
        id_document, member_from, member_to = relation
        return '\t'.join([
            label, id_domain, id_document,
            member_from.id_sentence, member_from.channel, str(member_from.indices), member_from.lemma,
            member_to.id_sentence, member_to.channel, str(member_to.indices), member_to.lemma,
            str(member_from.is_named_entity), str(member_to.is_named_entity)
        ])

    def make_map(self) -> [List, torch.tensor]:
        keys = []
        vectors = []

        for label, id_domain, relation in self.relations_loader.relations():
            id_document, member_from, member_to = relation

            if self._is_phrase_too_long(member_from) or self._is_phrase_too_long(member_to):
                continue

            ner_values = [member_from.is_named_entity, member_to.is_named_entity]

            key = self._make_key(label, id_domain, relation)
            vector = torch.FloatTensor(ner_values).unsqueeze(0)

            keys.append(key)
            vectors.append(vector)

        return keys, torch.cat(vectors)


@click.command()
@click.option('--relations-file', required=True, type=str,
              help='Path to relations file.')
@click.option('--output-paths', required=True, type=(str, str),
              metavar='ner.rel.keys ner.rel.pt',
              help='Paths for saving keys and map files.')
def main(relations_file, output_paths):
    relations_loader = RelationsLoader(relations_path=Path(relations_file))

    mapmaker = RelationsMapMaker(relations_loader)
    relations_keys, relations_vectors = mapmaker.make_map()

    keys_path, vectors_path = output_paths
    save_lines(Path(keys_path), relations_keys)
    save_tensor(Path(vectors_path), relations_vectors)


if __name__ == '__main__':
    main()
