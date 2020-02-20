#!/usr/bin/env python3.6

from pathlib import Path
from typing import List, Iterator

import click

from data.scripts.entities import Member, Relation
from data.scripts.relations import RelationsLoader
from data.scripts.utils.io import save_json


class SPERTMapper:
    ENTITY_TYPE_MAP = {
        'BRAND_NAME': 'Brand',
        'PRODUCT_NAME': 'Product',
    }

    RELATION_TYPE_MAP = {
        'BRAND_NAME PRODUCT_NAME': 'Brand-Product',
        'PRODUCT_NAME BRAND_NAME': 'Product-Brand'
    }

    def __init__(self, relations_loader: RelationsLoader):
        self.relations_loader = relations_loader

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

    def map_relations(self, relation: Relation) -> List[dict]:
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

    def filter_map(self) -> Iterator[dict]:
        for label, id_domain, relation in self.relations_loader.relations():
            in_relation = label == 'in_relation'
            in_same_context = relation.member_from.context == relation.member_to.context

            if in_relation and in_same_context:
                yield self.map(relation)


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--output-path', required=True, type=str,
              help='Paths for saving SPERT json file.')
def main(input_path, output_path):
    relations_loader = RelationsLoader(Path(input_path))
    mapper = SPERTMapper(relations_loader)

    documents = mapper.filter_map()

    save_json(documents, Path(output_path))


if __name__ == '__main__':
    main()
