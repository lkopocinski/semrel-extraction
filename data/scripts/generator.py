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
    relations = {}
    relidxs = {}
    for relation in document.relations():
        if is_ner_relation(relation) and is_in_channel(relation, channels):
            sent_id_from = relation.rel_from().sentence_id()
            sent_id_to = relation.rel_to().sentence_id()

            element_from = get_relation_element(relation.rel_from(), sentences)
            element_to = get_relation_element(relation.rel_to(), sentences)

            indices_from = tuple(element_from.indices)
            indices_to = tuple(element_to.indices)

            # We consider relation data in both ways
            relations[
                ((sent_id_from, indices_from), (sent_id_to, indices_to))
            ] = (relation, element_from.context, element_to.context)
            relations[
                ((sent_id_to, indices_to), (sent_id_from, indices_from))
            ] = (relation, element_to.context, element_from.context)

            # We add the same metadata to all indices in phrase
            for idx_from in indices_from:
                relidxs[(sent_id_from, idx_from)] = (
                    indices_from, element_from.channel, element_from.ne)
            for idx_to in indices_to:
                relidxs[(sent_id_to, idx_to)] = (
                    indices_to, element_to.channel, element_to.ne)

    for rel, rel_value in relations.items():
        relation, context_from, context_to = rel_value
        ((sent_id_from, indices_from), (sent_id_to, indices_to)) = rel

        nouns_indices_from = get_nouns_idx(sentences[sent_id_from])
        nouns_indices_to = get_nouns_idx(sentences[sent_id_to])

        if sent_id_from == sent_id_to:
            nouns_indices_pairs = permutations(nouns_indices_from, 2)
        else:
            nouns_indices_pairs = product(nouns_indices_from, nouns_indices_to)

        for idx_from, idx_to in nouns_indices_pairs:
            try:
                _f_idxs, _f_channel_name, _f_ne = relidxs[
                    (sent_id_from, idx_from)]
            except KeyError:
                _f_idxs = None
                _f_channel_name = ''
                _f_ne = False
                pass

            try:
                _t_idxs, _t_channel_name, _t_ne = relidxs[(sent_id_to, idx_to)]
            except KeyError:
                _t_idxs = None
                _t_channel_name = ''
                _t_ne = False
                pass

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
            rel = Relation(id_document, element_from, element_to)

            lines.append(f'no_relation\t{id_domain}\t{rel}')

    return lines
