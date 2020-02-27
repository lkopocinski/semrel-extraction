#!/usr/bin/env python3.6
import json
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import List, Iterator, NamedTuple, Set, Tuple

import click

from data.scripts.entities import Member, Relation
from data.scripts.relations import RelationsLoader
from data.scripts.utils.io import save_json


class Indices(NamedTuple):
    train: List[int]
    valid: List[int]
    test: List[int]


class Relations(NamedTuple):
    train: List[str] = []
    valid: List[str] = []
    test: List[str] = []


class SPERTEntity(NamedTuple):
    entity_type: str
    start: int
    end: int

    def to_dict(self):
        return {"type": self.entity_type, "start": self.start, "end": self.end}


class SPERTRelation(NamedTuple):
    tokens: List[str]
    head: SPERTEntity
    tail: SPERTEntity
    relation_type: str


class SPERTDocRelation(NamedTuple):
    head: int
    tail: int
    relation_type: str

    def to_dict(self):
        return {'type': self.relation_type, 'head': self.head, 'tail': self.tail}


class SPERTDocument(NamedTuple):
    tokens: List[str]
    entities: List[SPERTEntity]
    relations: Set[SPERTDocRelation]

    def to_dict(self):
        return {
            'tokens': self.tokens,
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations]
        }


class BrandProductSPERTMapper(ABC):
    ENTITY_TYPE_MAP = {
        'BRAND_NAME': 'Brand',
        'PRODUCT_NAME': 'Product',
    }

    RELATION_TYPE_MAP = {
        'BRAND_NAME PRODUCT_NAME': 'Brand-Product',
        'PRODUCT_NAME BRAND_NAME': 'Product-Brand'
    }

    def map(self, relation: Relation) -> SPERTRelation:
        tokens = self.map_tokens(relation)
        head, tail = self.map_entities(relation)
        relation_type = self._map_relation_type(relation)
        return SPERTRelation(tokens, head, tail, relation_type)

    def _map_relation_type(self, relation: Relation) -> str:
        relation_key = f'{relation.member_from.channel} {relation.member_to.channel}'
        return self.RELATION_TYPE_MAP[relation_key]

    def map_entities(self, relation: Relation) -> Tuple[SPERTEntity, SPERTEntity]:
        pass

    def map_tokens(self, relation: Relation) -> List[str]:
        pass


class InSentenceSPERTMapper(BrandProductSPERTMapper):

    def map_tokens(self, relation: Relation):
        return relation.member_from.context

    def _map_entity(self, member: Member) -> SPERTEntity:
        entity_type = self.ENTITY_TYPE_MAP[member.channel]
        start = member.indices[0]
        end = member.indices[-1] + 1
        return SPERTEntity(entity_type, start, end)

    def map_entities(self, relation: Relation) -> Tuple[SPERTEntity, SPERTEntity]:
        entity_from = self._map_entity(relation.member_from)
        entity_to = self._map_entity(relation.member_to)
        return entity_from, entity_to


class BetweenSentencesSPERTMapper(BrandProductSPERTMapper):

    def map_tokens(self, relation: Relation):
        return relation.member_from.context + relation.member_to.context

    def _map_entity(self, member: Member, shift: int = 0) -> SPERTEntity:
        entity_type = self.ENTITY_TYPE_MAP[member.channel]
        start = shift + member.indices[0]
        end = shift + member.indices[-1] + 1
        return SPERTEntity(entity_type, start, end)

    def map_entities(self, relation: Relation) -> Tuple[SPERTEntity, SPERTEntity]:
        member_from_context_len = len(relation.member_from.context)
        entity_from = self._map_entity(relation.member_from)
        entity_to = self._map_entity(relation.member_to, shift=member_from_context_len)
        return entity_from, entity_to


def load_indices(indices_file: Path) -> Indices:
    with indices_file.open('r', encoding='utf-8') as file:
        indices = json.load(file)
        return Indices(
            train=indices['train'],
            valid=indices['valid'],
            test=indices['test']
        )


def load_relations(indices: Indices, relations_loader: RelationsLoader) -> Iterator[Relation]:
    relations = Relations()

    for index, (label, _, relation) in enumerate(relations_loader.relations()):
        if label == 'in_relation':
            if index in indices.train:
                relations.train.append(relation)
            elif index in indices.valid:
                relations.valid.append(relation)
            elif index in indices.test:
                relations.test.append(relation)

    return relations


def map_relations(relations: Iterator[Relation],
                  in_sentence_mapper: InSentenceSPERTMapper,
                  between_sentence_mapper: BetweenSentencesSPERTMapper):
    documents = defaultdict(SPERTDocument)

    for relation in relations:
        id_document, member_from, member_to = relation
        in_same_context = member_from.id_sentence == member_to.id_sentence

        id_from = relation.member_from.id_sentence
        id_to = relation.member_from.id_sentence
        key = f'{id_document}-{id_from}-{id_to}'

        if in_same_context:
            spert_relation = in_sentence_mapper.map(relation)
        else:
            spert_relation = between_sentence_mapper.map(relation)

        document = documents[key]
        document.tokens = spert_relation.tokens

        if spert_relation.head not in document.entities:
            document.entities.append(spert_relation.head)

        if spert_relation.tail not in document.entities:
            document.entities.append(spert_relation.tail)

        index_from = document.entities.index(spert_relation.head)
        index_to = document.entities.index(spert_relation.tail)

        document.relations.add(SPERTDocRelation(index_from, index_to, spert_relation.relation_type))

    return documents


@click.command()
@click.option('--input-path', required=True, type=str,
              help='Path to relations file.')
@click.option('--indices-file', required=True, type=str,
              help='Path to indices file.')
@click.option('--output-path', required=True, type=str,
              help='Paths for saving SPERT json file.')
def main(input_path, indices_file, output_path):
    relations_loader = RelationsLoader(Path(input_path))

    indices = load_indices(Path(indices_file))
    relations = load_relations(indices, relations_loader)

    in_sentence_mapper = InSentenceSPERTMapper()
    between_sentence_mapper = BetweenSentencesSPERTMapper()

    documents = map_relations(relations, in_sentence_mapper, between_sentence_mapper)
    documents = [document.to_dict() for document in documents]

    save_json(documents, Path(output_path))

if __name__ == '__main__':
    main()
