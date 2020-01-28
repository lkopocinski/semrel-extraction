#!/usr/bin/env python3.6

from itertools import permutations, product

from data.scripts.utils.corpus import Document, Element


class RelationsGenerator:

    def __init__(self, document: Document):
        self._document = document

        self._relation_tokens_indices = None

    def generate_positive(self,
                          channels: tuple = (('BRAND_NAME', 'PRODUCT_NAME'), ('PRODUCT_NAME', 'BRAND_NAME'))):
        for relation in self._document.relations:
            if relation.is_ner() and relation.channels in channels:
                member_from, member_to = relation.get_members(self._document._sentence_dict)
                yield Relation(self._document.id, member_from, member_to)

    def generate_negative(self,
                          channels: tuple = (('BRAND_NAME', 'PRODUCT_NAME'), ('PRODUCT_NAME', 'BRAND_NAME'))):
        relations = []

        for relation in self._document.relations:
            if relation.is_ner() and relation.channels in channels:
                member_from, member_to = relation.get_members(self._document._sentence_dict)

                relations.append(Relation(self._document.id, member_from, member_to))

                self._map_indices_to_relation(member_from)
                self._map_indices_to_relation(member_to)

        for _, member_from, member_to in relations:
            nouns_indices_pairs = self._get_nouns_indices_pairs(member_from, member_to)

            for idx_from, idx_to in nouns_indices_pairs:
                _member_from = self._relation_tokens_indices.get(
                    (member_from.sent_id, idx_from), None)

                _member_to = self._relation_tokens_indices.get(
                    (member_to.sent_id, idx_to), None)

                if _member_from and _member_to and Relation(self._document.id, _member_from, _member_to) in relations:
                    continue

                lemma_from = self._document._sentence_dict[member_from.sent_id].lemmas[idx_from]
                lemma_to = self._document._sentence_dict[member_to.sent_id].lemmas[idx_to]

                member_from = Element(
                    member_from.sent_id,
                    lemma_from,
                    _member_from.channel if _member_from else '',
                    _member_from.ne if _member_from else False,
                    (idx_from,),
                    member_from.context
                )
                member_to = Element(
                    member_to.sent_id,
                    lemma_to,
                    _member_to.channel if _member_to else '',
                    _member_from.ne if _member_from else False,
                    (idx_to,),
                    member_to.context
                )

                id_document = self._document.id
                yield Relation(id_document, member_from, member_to)

    def _map_indices_to_relation(self, member):
        for idx_from in member.indices:
            self._relation_tokens_indices[(member.sent_id, idx_from)] = member

    def _get_nouns_indices_pairs(self, member_from, member_to):
        sent_id_from = member_from.sent_id
        sent_id_to = member_to.sent_id

        nouns_indices_from = self._document._sentence_dict[sent_id_from].noun_indices
        nouns_indices_to = self._document._sentence_dict[sent_id_to].noun_indices

        if sent_id_from == sent_id_to:
            nouns_indices_pairs = permutations(nouns_indices_from, 2)
        else:
            nouns_indices_pairs = product(nouns_indices_from, nouns_indices_to)

        return nouns_indices_pairs
