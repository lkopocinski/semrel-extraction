#!/usr/bin/env python3.6
import csv
from itertools import permutations, product
from pathlib import Path
from typing import Iterator, List, Dict

from data.scripts.entities import Relation
from data.scripts.utils.corpus import Document, Member


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


class RelationsLoader:

    def __init__(self, relations_path: Path):
        self._relations_path_csv = relations_path

    def relations(self):
        with self._relations_path_csv.open('r', newline='', encoding='utf-8') as file_csv:
            reader_csv = csv.DictReader(file_csv, delimiter='\t')
            for line_dict in reader_csv:
                label = line_dict['label']
                id_domain = line_dict['id_domain']
                relation = self._parse_relation(line_dict)

                yield label, id_domain, relation

    def _filter_relations(self, filter_label: str) -> Dict[int, Relation]:
        """Experimental feature"""

        relations_dict = {}
        index = 0
        for label, _, relation in self.relations():
            if len(relation.member_from.indices) > 5 or len(relation.member_to.indices) > 5:
                continue

            if label != filter_label:
                index += 1
                continue


            relations_dict[index] = relation
            index += 1

        return relations_dict

    @staticmethod
    def _parse_relation(relation_dict: dict) -> Relation:
        id_document = relation_dict['id_document']
        member_from = Member(
            id_sentence=relation_dict['id_sentence_from'],
            lemma=relation_dict['lemma_from'],
            channel=relation_dict['channel_from'],
            is_named_entity=eval(relation_dict['is_named_entity_from']),
            indices=eval(relation_dict['indices_from']),
            context=eval(relation_dict['context_from'])
        )
        member_to = Member(
            id_sentence=relation_dict['id_sentence_to'],
            lemma=relation_dict['lemma_to'],
            channel=relation_dict['channel_to'],
            is_named_entity=eval(relation_dict['is_named_entity_to']),
            indices=eval(relation_dict['indices_to']),
            context=eval(relation_dict['context_to'])
        )

        return Relation(id_document, member_from, member_to)
