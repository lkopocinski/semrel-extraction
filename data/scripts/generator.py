#!/usr/bin/env python3.6

from itertools import permutations, product
from pathlib import Path
from typing import Iterator

from data.scripts.models import Relation
from data.scripts.utils.corpus import id_to_sent_dict, is_ner_relation, \
    is_in_channel, get_relation_element, get_nouns_idx, \
    get_lemma, relations_documents_gen, get_document_dir, \
    get_document_file_name


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

    id_document = get_document_file_name(document)

    for relation in document.relations():
        if is_ner_relation(relation) and is_in_channel(relation, channels):
            element_from = get_relation_element(relation.rel_from(), sentences)
            element_to = get_relation_element(relation.rel_to(), sentences)

            # We add the same metadata to all indices in phrase
            map_indices_to_relation(element_from, relation_tokens_indices)
            map_indices_to_relation(element_to, relation_tokens_indices)

            # We consider relation data in both ways
            relations_list.extend([
                Relation(id_document, element_from, element_to),
                Relation(id_document, element_to, element_from),
            ])

    for relation in relations_list:
        element_from = relation.source
        element_to = relation.dest

        nouns_indices_pairs = get_nouns_indices_pairs(relation, sentences)

        for idx_from, idx_to in nouns_indices_pairs:
            _f_idxs, _f_channel_name, _f_ne = relation_tokens_indices.get(
                (element_from.sent_id, idx_from), (None, '', False))

            _t_idxs, _t_channel_name, _t_ne = relation_tokens_indices.get(
                (element_to.sent_id, idx_to), (None, '', False))

            # If two nouns are part of phrases in relation
            if _t_idxs and _f_idxs:
                if ((element_to.sent_id, _t_idxs),
                    (element_from.sent_id, _f_idxs)) in relations_list:
                    continue

            f_lemma = get_lemma(sentences[element_from.sent_id], idx_from)
            t_lemma = get_lemma(sentences[element_to.sent_id], idx_to)

            element_from = Relation.Element(
                element_from.sent_id, f_lemma, _f_channel_name,
                _f_ne, [idx_from], element_from.context
            )
            element_to = Relation.Element(
                element_to.sent_id, t_lemma, _t_channel_name,
                _t_ne, [idx_to], element_to.context
            )

            id_domain = get_document_dir(document)
            id_document = get_document_file_name(document)
            relation = Relation(id_document, element_from, element_to)

            lines.append(f'no_relation\t{id_domain}\t{relation}')

    return lines


def map_indices_to_relation(element, tokens_indices):
    for idx_from in element.indices:
        tokens_indices[(element.sent_id, idx_from)] = element


def get_nouns_indices_pairs(relation, sentences):
    sent_id_from = relation.source.sent_id
    sent_id_to = relation.dest.sent_id

    nouns_indices_from = get_nouns_idx(sentences[sent_id_from])
    nouns_indices_to = get_nouns_idx(sentences[sent_id_to])

    if sent_id_from == sent_id_to:
        nouns_indices_pairs = permutations(nouns_indices_from, 2)
    else:
        nouns_indices_pairs = product(nouns_indices_from, nouns_indices_to)

    return nouns_indices_pairs
