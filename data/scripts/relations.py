#!/usr/bin/env python3.6

from itertools import permutations, product
from typing import Iterator, List

from data.scripts.utils.corpus import Document, Member
from entities import Relation


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
            nouns_indices_pairs = self._get_nouns_indices_pairs(member_from, member_to)

            for index_from, index_to in nouns_indices_pairs:
                _member_from = self._tokens_indices_to_member.get(
                    (member_from.id_sentence, index_from), None)

                _member_to = self._tokens_indices_to_member.get(
                    (member_to.id_sentence, index_to), None)

                if self._are_in_relation(_member_from, _member_to, relations):
                    continue

                lemma_from = self._document.get_sentence(member_from.id_sentence).lemmas[index_from]
                lemma_to = self._document.get_sentence(member_to.id_sentence).lemmas[index_to]

                member_from = Member(
                    member_from.id_sentence,
                    lemma_from,
                    _member_from.channel if _member_from else '',
                    _member_from.is_named_entity if _member_from else False,
                    (index_from,),
                    member_from.context
                )
                member_to = Member(
                    member_to.id_sentence,
                    lemma_to,
                    _member_to.channel if _member_to else '',
                    _member_from.is_named_entity if _member_from else False,
                    (index_to,),
                    member_to.context
                )

                _ = self._document.id
                yield Relation(_, member_from, member_to)

    def _map_tokens_indices_to_member(self, member: Member):
        for index in member.indices:
            self._tokens_indices_to_member[(member.id_sentence, index)] = member

    def _get_nouns_indices_pairs(self, member_from: Member, member_to: Member):
        sent_id_from = member_from.id_sentence
        sent_id_to = member_to.id_sentence

        nouns_indices_from = self._document.get_sentence(sent_id_from).noun_indices
        nouns_indices_to = self._document.get_sentence(sent_id_to).noun_indices

        if sent_id_from == sent_id_to:
            nouns_indices_pairs = permutations(nouns_indices_from, r=2)
        else:
            nouns_indices_pairs = product(nouns_indices_from, nouns_indices_to)

        return nouns_indices_pairs

    def _are_in_relation(self, member_from: Member, member_to: Member, relations: List):
        return member_from and member_to and (
            (Relation(self._document.id, member_from, member_to) in relations
             or Relation(self._document.id, member_to, member_from) in relations))
