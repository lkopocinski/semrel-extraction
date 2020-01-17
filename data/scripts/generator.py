#!/usr/bin/env python3.6

from itertools import permutations, product
from pathlib import Path
from typing import Iterator

from data.scripts.models import Relation
from data.scripts.utils.corpus import id_to_sent_dict, is_ner_relation, \
    is_in_channel, get_relation_element, get_nouns_idx, \
    get_lemma, relations_documents_gen, get_document_dir, \
    get_document_file_name, Document


class RelationsGenerator:

    def __init__(self, document: Document):
        self._document = document

    def generate_negative(self, channels: tuple):
        for relation in self._document.relations:
            if relation.is_ner() and relation.channels in (('BRAND_NAME', 'PRODUCT_NAME'), ('PRODUCT_NAME', 'BRAND_NAME')):




def generate(relation_files: Iterator[Path],
             channels: tuple = ('BRAND_NAME', 'PRODUCT_NAME')):
    for document in relations_documents_gen(relation_files):
        sentences = id_to_sent_dict(document)

        # yield generate_positive(document, sentences, channels)
        yield generate_negative(document, sentences, channels)



def generate_positive(document, sentences, channels):
    lines = []
    for relation in document.relations():
        if is_ner_relation(relation) and is_in_channel(relation, channels):
            element_from = get_relation_element(relation.rel_from(), sentences)
            element_to = get_relation_element(relation.rel_to(), sentences)

            if element_from and element_to:
                id_domain = get_document_dir(document)
                id_document = get_document_file_name(document)

                rel = Relation(id_document, element_from, element_to)
                lines.append(f'in_relation\t{id_domain}\t{rel}')
    return lines



def generate_negative(document, sentences, channels):
    lines = []

    relations_list = []
    relation_tokens_indices = {}

    for relation in document.relations():
        if is_ner_relation(relation) and is_in_channel(relation, channels):
            element_from = get_relation_element(relation.rel_from(), sentences)
            element_to = get_relation_element(relation.rel_to(), sentences)

            # We add the same metadata to all indices in phrase
            map_indices_to_relation(element_from, relation_tokens_indices)
            map_indices_to_relation(element_to, relation_tokens_indices)

            # We consider relation data in both ways
            relations_list.extend([
                (element_from, element_to),
                (element_to, element_from),
            ])

    for element_from, element_to in relations_list:
        nouns_indices_pairs = get_nouns_indices_pairs(element_from, element_to, sentences)

        for idx_from, idx_to in nouns_indices_pairs:
            _element_from = relation_tokens_indices.get(
                (element_from.sent_id, idx_from), None)

            _element_to = relation_tokens_indices.get(
                (element_to.sent_id, idx_to), None)

            # If two nouns are part of phrases in relation
            if _element_from and _element_to:
                if are_in_relation(_element_to, _element_from, relations_list):
                    continue

            f_lemma = get_lemma(sentences[element_from.sent_id], idx_from)
            t_lemma = get_lemma(sentences[element_to.sent_id], idx_to)

            element_from = Relation.Element(
                element_from.sent_id,
                f_lemma,
                _element_from.channel if _element_from else '',
                _element_from.ne if _element_from else False,
                [idx_from],
                element_from.context
            )
            element_to = Relation.Element(
                element_to.sent_id,
                t_lemma,
                _element_to.channel if _element_to else '',
                _element_from.ne if _element_from else False,
                [idx_to],
                element_to.context
            )

            id_domain = get_document_dir(document)
            id_document = get_document_file_name(document)
            relation = Relation(id_document, element_from, element_to)

            lines.append(f'no_relation\t{id_domain}\t{relation}')

    return lines


def are_in_relation(element_from, element_to, relations_list):
    # Must be hashed or data classes like
    # This is not gonna work unless is set
    return (element_from, element_to) in relations_list


def map_indices_to_relation(element, tokens_indices):
    for idx_from in element.indices:
        tokens_indices[(element.sent_id, idx_from)] = element


def get_nouns_indices_pairs(element_from, element_to, sentences):
    sent_id_from = element_from.sent_id
    sent_id_to = element_to.sent_id

    nouns_indices_from = get_nouns_idx(sentences[sent_id_from])
    nouns_indices_to = get_nouns_idx(sentences[sent_id_to])

    if sent_id_from == sent_id_to:
        nouns_indices_pairs = permutations(nouns_indices_from, 2)
    else:
        nouns_indices_pairs = product(nouns_indices_from, nouns_indices_to)

    return nouns_indices_pairs
