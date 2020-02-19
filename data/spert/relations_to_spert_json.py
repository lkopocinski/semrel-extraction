#!/usr/bin/env python3.6

import json
from pathlib import Path
from typing import List, Iterator

import click

from data.scripts.combine_vectors import RelationsLoader
from data.scripts.entities import Member, Relation


class SPERTMapper:
    ENTITY_TYPE_MAP = {
        'BRAND_NAME': 'Brand',
        'PRODUCT_NAME': 'Product',
    }

    RELATION_TYPE_MAP = {
        'BRAND_NAME PRODUCT_NAME': 'Brand-Product',
        'PRODUCT_NAME BRAND_NAME': 'Product-Brand'
    }

    def map_tokens(self, relation: Relation):
        return relation.member_from.context

    def map_entity(self, member: Member) -> dict:
        entity_type = self.ENTITY_TYPE_MAP[member.channel]
        start = member.indices[0]
        end = member.indices[-1] + 1

        return {"type": entity_type, "start": start, "end": end}

    def map_entities(self, relation: Relation) -> List[dict]:
        return [self.map_entity(relation.member_from),
                self.map_entity(relation.member_to)]

    def map_relations(self, relation: Relation):
        relation_key = f'{relation.member_from.channel} {relation.member_to.channel}'
        relation_type = self.RELATION_TYPE_MAP[relation_key]
        relation_dict = {'type': relation_type, 'head': 0, 'tail': 1}

        return [relation_dict]

    def map(self, relation: Relation) -> dict:
        return {
            'tokens': self.map_tokens(relation),
            'entities': self.map_entities(relation),
            'relations': self.map_relations(relation)
        }


def filter_relations(relations_loader: RelationsLoader) -> Iterator[Relation]:
    for label, id_domain, relation in relations_loader.relations():
        in_relation = (label == 'in_relation')
        in_same_context = (relation.member_from.context == relation.member_to.context)

        if in_relation and in_same_context:
            yield relation


def save_json(documents, save_path: Path):
    with save_path.open("w", encoding='utf-8') as file:
        json.dump(documents, file)


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relation corpora files index.')
@click.option('--output-path', required=True, type=str,
              help='Paths for saving SPERT json file.')
def main(input_path: str, output_path: str):
    mapper = SPERTMapper()
    relations_loader = RelationsLoader(Path(input_path))

    documents = [
        mapper.map(relation)
        for relation in filter_relations(relations_loader)
    ]

    save_json(documents, Path(output_path))


if __name__ == '__main__':
    main()
