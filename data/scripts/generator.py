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
    # relations = {}
    relation_tokens_indices = {}

    id_document = get_document_file_name(document)

    for relation in document.relations():
        if is_ner_relation(relation) and is_in_channel(relation, channels):
            element_from = get_relation_element(relation.rel_from(), sentences)
            element_to = get_relation_element(relation.rel_to(), sentences)

            # We consider relation data in both ways
            relations_list.extend([
                Relation(id_document, element_from, element_to),
                Relation(id_document, element_to, element_from),
                ])
            # relations[(
            #     (element_from.sent_id, element_from.indices),
            #     (element_to.sent_id, element_to.indices)
            # )] = (element_from.context, element_to.context)
            # relations[(
            #     (element_to.sent_id, element_to.indices),
            #     (element_from.sent_id, element_from.indices)
            # )] = (element_to.context, element_from.context)

            # We add the same metadata to all indices in phrase
            for idx_from in element_from.indices:
                relation_tokens_indices[(element_from.sent_id, idx_from)] = (
                    element_from.indices, element_from.channel, element_from.ne
                )

            for idx_to in element_to.indices:
                relation_tokens_indices[(element_to.sent_id, idx_to)] = (
                    element_to.indices, element_to.channel, element_to.ne
                )

    for relation, relation_contexts in relations.items():
        ((sent_id_from, indices_from), (sent_id_to, indices_to)) = relation
        context_from, context_to = relation_contexts

        nouns_indices_pairs = get_nouns_indices_pairs(
            sent_id_from, sent_id_to, sentences)

        for idx_from, idx_to in nouns_indices_pairs:
            _f_idxs, _f_channel_name, _f_ne = relation_tokens_indices.get(
                (sent_id_from, idx_from), (None, '', False))

            _t_idxs, _t_channel_name, _t_ne = relation_tokens_indices.get(
                (sent_id_to, idx_to), (None, '', False))

            # If two nouns are part of phrases in relation
            if _t_idxs \
                    and _f_idxs \
                    and ((sent_id_to, _t_idxs), (sent_id_from, _f_idxs)) \
                    in relations:
                continue

            f_lemma = get_lemma(sentences[sent_id_from], idx_from)
            t_lemma = get_lemma(sentences[sent_id_to], idx_to)

            element_from = Relation.Element(
                sent_id_from, f_lemma, _f_channel_name,
                _f_ne, [idx_from], context_from
            )
            element_to = Relation.Element(
                sent_id_to, t_lemma, _t_channel_name,
                _t_ne, [idx_to], context_to
            )

            id_domain = get_document_dir(document)
            id_document = get_document_file_name(document)
            relation = Relation(id_document, element_from, element_to)

            lines.append(f'no_relation\t{id_domain}\t{relation}')

    return lines


def get_nouns_indices_pairs(sent_id_from, sent_id_to, sentences):
    nouns_indices_from = get_nouns_idx(sentences[sent_id_from])
    nouns_indices_to = get_nouns_idx(sentences[sent_id_to])

    if sent_id_from == sent_id_to:
        nouns_indices_pairs = permutations(nouns_indices_from, 2)
    else:
        nouns_indices_pairs = product(nouns_indices_from, nouns_indices_to)

    return nouns_indices_pairs
