from typing import NamedTuple, List, Set


class Indices(NamedTuple):
    train: List[int]
    valid: List[int]
    test: List[int]


class SPERTEntity(NamedTuple):
    entity_type: str
    start: int
    end: int

    def to_dict(self):
        return {'type': self.entity_type,
                'start': self.start,
                'end': self.end}


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
        return {'type': self.relation_type,
                'head': self.head,
                'tail': self.tail}


class SPERTDocument:

    def __init__(
            self,
            tokens: List[str] = None,
            entities: List[SPERTEntity] = None,
            relations: Set[SPERTDocRelation] = None
    ):
        self.tokens = tokens or []
        self.entities = entities or []
        self.relations = relations or set()

    def to_dict(self):
        return {
            'tokens': self.tokens,
            'entities': [entity.to_dict() for entity in self.entities],
            'relations': [relation.to_dict() for relation in self.relations]
        }
