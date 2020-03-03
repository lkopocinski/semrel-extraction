
from abc import ABC, abstractmethod
from typing import Tuple, List

from entities import Relation, Member
from spert.entities import SPERTEntity, SPERTRelation


class BrandProductSPERTMapper(ABC):
    ENTITY_TYPE_MAP = {
        'BRAND_NAME': 'Brand',
        'PRODUCT_NAME': 'Product',
    }

    RELATION_TYPE_MAP = {
        'BRAND_NAME-PRODUCT_NAME': 'Brand-Product',
        'PRODUCT_NAME-BRAND_NAME': 'Product-Brand'
    }

    def map(self, relation: Relation) -> SPERTRelation:
        tokens = self.map_tokens(relation)
        head, tail = self.map_entities(relation)
        relation_type = self._map_relation_type(relation)
        return SPERTRelation(tokens, head, tail, relation_type)

    def _map_relation_type(self, relation: Relation) -> str:
        relation_key = f'{relation.member_from.channel}-{relation.member_to.channel}'
        return self.RELATION_TYPE_MAP[relation_key]

    @abstractmethod
    def map_entities(self, relation: Relation) -> Tuple[SPERTEntity, SPERTEntity]:
        pass

    @abstractmethod
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
