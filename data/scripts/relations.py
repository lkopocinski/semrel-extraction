#!/usr/bin/env python3.6

import csv
from itertools import permutations, product
from pathlib import Path
from typing import Iterator, List, Dict

import torch

import data.scripts.constant as constant
from data.scripts.constant import RelationHeader as rh
from data.scripts.entities import Relation
from data.scripts.keys import make_token_key_member, make_relation_key
from data.scripts.maps import VectorsMap
from data.scripts.max_pool import max_pool_relation_vectors
from data.scripts.utils.corpus import Document, Member


def is_phrase_too_long(member: Member) -> bool:
    return len(member.indices) > constant.PHRASE_LENGTH_LIMIT


class RelationsGenerator:

    def __init__(self, document: Document):
        self._document = document

        self._tokens_indices_to_member = {}

    def generate_positive(self, channels: tuple) -> Iterator[Relation]:
        for relation in self._document.relations:
            if relation.is_ner and relation.channels in channels:
                member_from, member_to = relation.get_members()
                yield Relation(self._document.id, member_from, member_to)

    def generate_negative(self, channels: tuple):
        relations = []

        for relation in self.generate_positive(channels):
            self._map_tokens_indices_to_member(relation.member_from)
            self._map_tokens_indices_to_member(relation.member_to)
            relations.append(relation)

        for _, member_from, member_to in relations:
            nouns_indices_pairs = self._get_nouns_indices_pairs(
                member_from, member_to
            )

            for index_from, index_to in nouns_indices_pairs:
                _member_from = self._tokens_indices_to_member.get(
                    (member_from.id_sentence, index_from), None)

                _member_to = self._tokens_indices_to_member.get(
                    (member_to.id_sentence, index_to), None)

                if self._are_in_relation(_member_from, _member_to, relations):
                    continue

                sentence_from = self._document.get_sentence(
                    member_from.id_sentence
                )
                sentence_to = self._document.get_sentence(
                    member_to.id_sentence
                )

                lemma_from = sentence_from.lemmas[index_from]
                lemma_to = sentence_to.lemmas[index_to]

                __member_from = Member(
                    id_sentence=member_from.id_sentence,
                    lemma=lemma_from,
                    channel=_member_from.channel if _member_from else '',
                    is_named_entity=_member_from.is_named_entity if _member_from else False,
                    indices=(index_from,),
                    context=member_from.context
                )
                __member_to = Member(
                    id_sentence=member_to.id_sentence,
                    lemma=lemma_to,
                    channel=_member_to.channel if _member_to else '',
                    is_named_entity=_member_to.is_named_entity if _member_to else False,
                    indices=(index_to,),
                    context=member_to.context
                )

                yield Relation(self._document.id, __member_from, __member_to)

    def _map_tokens_indices_to_member(self, member: Member):
        for index in member.indices:
            self._tokens_indices_to_member[(member.id_sentence, index)] = member

    def _get_nouns_indices_pairs(
            self, member_from: Member, member_to: Member
    ):
        sent_id_from = member_from.id_sentence
        sent_id_to = member_to.id_sentence

        nouns_indices_from = self._document.get_sentence(
            sent_id_from).noun_indices
        nouns_indices_to = self._document.get_sentence(sent_id_to).noun_indices

        if sent_id_from == sent_id_to:
            nouns_indices_pairs = permutations(nouns_indices_from, r=2)
        else:
            nouns_indices_pairs = product(nouns_indices_from, nouns_indices_to)

        return nouns_indices_pairs

    def _are_in_relation(
            self, member_from: Member, member_to: Member, relations: List
    ):
        return member_from and member_to and (
            (Relation(self._document.id, member_from, member_to) in relations
             or Relation(self._document.id, member_to,
                         member_from) in relations))


class RelationsLoader:

    def __init__(self, relations_path: Path):
        self._relations_path_csv = relations_path

    def relations(self):
        with self._relations_path_csv.open(
                'r', newline='', encoding='utf-8'
        ) as file_csv:
            reader_csv = csv.DictReader(file_csv, delimiter='\t')
            for line_dict in reader_csv:
                label = line_dict[rh.LABEL]
                id_domain = line_dict[rh.DOMAIN]
                relation = self._parse_relation(line_dict)

                yield label, id_domain, relation

    def _filter_relations(self, filter_label: str) -> Dict[int, Relation]:
        """Experimental feature"""
        relations_dict = {}
        index = 0
        for label, _, relation in self.relations():
            if is_phrase_too_long(relation.member_from) \
                    or is_phrase_too_long(relation.member_to):
                continue

            if label == filter_label:
                relations_dict[index] = relation

            index += 1

        return relations_dict

    @staticmethod
    def _parse_relation(relation_dict: Dict) -> Relation:
        id_document = relation_dict[rh.DOCUMENT]
        member_from = Member(
            id_sentence=relation_dict[rh.SENTENCE_FROM],
            lemma=relation_dict[rh.LEMMA_FROM],
            channel=relation_dict[rh.CHANNEL_FROM],
            is_named_entity=eval(relation_dict[rh.NAMED_ENTITY_FROM]),
            indices=eval(relation_dict[rh.INDICES_FROM]),
            context=eval(relation_dict[rh.CONTEXT_FROM])
        )
        member_to = Member(
            id_sentence=relation_dict[rh.SENTENCE_TO],
            lemma=relation_dict[rh.LEMMA_TO],
            channel=relation_dict[rh.CHANNEL_TO],
            is_named_entity=eval(relation_dict[rh.NAMED_ENTITY_TO]),
            indices=eval(relation_dict[rh.INDICES_TO]),
            context=eval(relation_dict[rh.CONTEXT_TO])
        )

        return Relation(id_document, member_from, member_to)


class RelationsEmbedder:

    def __init__(
            self, relations_loader: RelationsLoader, vectors_map: VectorsMap
    ):
        self.relations_loader = relations_loader

        self._keys = vectors_map.keys
        self._vectors = vectors_map.vectors

    def _get_vectors_indices(
            self, id_domain: str, id_document: str, member: Member
    ) -> List[int]:
        tokens_keys = [
            make_token_key_member(id_domain, id_document, member, id_token)
            for id_token in member.indices]

        return [self._keys[token_key] for token_key in tokens_keys]

    def _get_member_tensor(
            self, vectors_indices: List[int], max_size
    ) -> torch.Tensor:
        tensor = torch.zeros(1, max_size, self._vectors.shape[-1])
        vectors = self._vectors[vectors_indices]
        tensor[:, 0:vectors.shape[0], :] = vectors
        return tensor

    def embed(self) -> [List, torch.Tensor]:
        keys = []
        vectors = []

        for label, id_domain, relation in self.relations_loader.relations():
            id_document, member_from, member_to = relation

            if is_phrase_too_long(member_from) or is_phrase_too_long(member_to):
                continue

            key = make_relation_key(label, id_domain, relation)

            vectors_indices_from = self._get_vectors_indices(
                id_domain, id_document, member_from
            )
            vectors_indices_to = self._get_vectors_indices(
                id_domain, id_document, member_to
            )

            vectors_from = self._get_member_tensor(
                vectors_indices_from, max_size=constant.PHRASE_LENGTH_LIMIT
            )
            vectors_to = self._get_member_tensor(
                vectors_indices_to, max_size=constant.PHRASE_LENGTH_LIMIT
            )

            keys.append(key)
            vectors.append((vectors_from, vectors_to))

        max_pooled_tensor = max_pool_relation_vectors(vectors)
        return keys, max_pooled_tensor
