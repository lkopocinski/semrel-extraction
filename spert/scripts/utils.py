from typing import Dict, List

from entities import Relation


def in_same_context(relation: Relation) -> bool:
    id_from = relation.member_from.id_sentence
    id_to = relation.member_to.id_sentence
    return id_from == id_to


def make_relation_key(relation: Relation) -> str:
    id_document = relation.id_document
    id_from = relation.member_from.id_sentence
    id_to = relation.member_to.id_sentence
    return f'{id_document}-{id_from}-{id_to}'


def split_relations(
        indices: Dict, relations_dict: Dict
) -> Dict[str, List[Relation]]:
    train_relations = _split_relations(indices['train'], relations_dict)
    valid_relations = _split_relations(indices['valid'], relations_dict)
    test_relations = _split_relations(indices['test'], relations_dict)
    return {
        'train': train_relations,
        'valid': valid_relations,
        'test': test_relations
    }


def _split_relations(indices: List, relations_dict: Dict) -> List[Relation]:
    return [
        relation
        for index, relation in relations_dict.items()
        if index in indices
    ]
