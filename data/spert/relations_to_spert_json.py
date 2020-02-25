#!/usr/bin/env python3.6
from abc import ABC
from pathlib import Path
from typing import List, Dict

import click

from data.scripts.entities import Member, Relation
from data.scripts.relations import RelationsLoader
from data.scripts.utils.io import save_json


class BrandProductSPERTMapper(ABC):
    ENTITY_TYPE_MAP = {
        'BRAND_NAME': 'Brand',
        'PRODUCT_NAME': 'Product',
    }

    RELATION_TYPE_MAP = {
        'BRAND_NAME PRODUCT_NAME': 'Brand-Product',
        'PRODUCT_NAME BRAND_NAME': 'Product-Brand'
    }

    def map(self, relation: Relation) -> Dict:
        return {
            'tokens': self.map_tokens(relation),
            'entities': self.map_entities(relation),
            'relations': self.map_relations(relation)
        }

    def map_relations(self, relation: Relation) -> List[Dict]:
        relation_key = f'{relation.member_from.channel} {relation.member_to.channel}'
        relation_type = self.RELATION_TYPE_MAP[relation_key]
        relation_dict = {'type': relation_type, 'head': 0, 'tail': 1}
        return [relation_dict]

    def map_tokens(self, relation: Relation):
        pass

    def map_entities(self, relation: Relation):
        pass


class InSentenceSPERTMapper(BrandProductSPERTMapper):

    def map_tokens(self, relation: Relation):
        return relation.member_from.context

    def map_entity(self, member: Member) -> Dict:
        entity_type = self.ENTITY_TYPE_MAP[member.channel]
        start = member.indices[0]
        end = member.indices[-1] + 1
        return {"type": entity_type, "start": start, "end": end}

    def map_entities(self, relation: Relation) -> List[Dict]:
        entity_from = self.map_entity(relation.member_from)
        entity_to = self.map_entity(relation.member_to)
        return [entity_from, entity_to]


class BetweenSentencesSPERTMapper(BrandProductSPERTMapper):

    def map_tokens(self, relation: Relation):
        return relation.member_from.context + relation.member_to.context

    def map_entity(self, member: Member, shift: int = 0) -> Dict:
        entity_type = self.ENTITY_TYPE_MAP[member.channel]
        start = shift + member.indices[0]
        end = shift + member.indices[-1] + 1
        return {"type": entity_type, "start": start, "end": end}

    def map_entities(self, relation: Relation) -> List[Dict]:
        member_from_context_len = len(relation.member_from.context)
        entity_from = self.map_entity(relation.member_from)
        entity_to = self.map_entity(relation.member_to, shift=member_from_context_len)
        return [entity_from, entity_to]


def map_in_relation(relations_loader: RelationsLoader,
                    in_sentence_mapper: InSentenceSPERTMapper,
                    between_sentence_mapper: BetweenSentencesSPERTMapper):
    in_relations = (relation
                    for label, _, relation in relations_loader.relations()
                    if label == 'in_relation')

    for relation in in_relations:
        _, member_from, member_to = relation
        in_same_context = member_from.context == member_to.context

        if in_same_context:
            yield in_sentence_mapper.map(relation)
        else:
            yield between_sentence_mapper.map(relation)


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--output-path', required=True, type=str,
              help='Paths for saving SPERT json file.')
def main(input_path, output_path):
    relations_loader = RelationsLoader(Path(input_path))
    in_sentence_mapper = InSentenceSPERTMapper()
    between_sentence_mapper = BetweenSentencesSPERTMapper()

    documents = map_in_relation(relations_loader, in_sentence_mapper, between_sentence_mapper)

    save_json(documents, Path(output_path))


if __name__ == '__main__':
    main()
