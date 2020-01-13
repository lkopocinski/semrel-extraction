#!/usr/bin/env python3.6

from itertools import permutations, product
from pathlib import Path
from typing import List, Iterator

from data.scripts.models import Relation
from data.scripts.utils.corpus import id_to_sent_dict, is_ner_relation, \
    is_in_channel, get_relation_element, get_nouns_idx, \
    get_lemma, relations_documents_gen, get_document_dir, \
    get_document_file_name


def generate(relation_files: Iterator[Path],
             channels: tuple = ('BRAND_NAME', 'PRODUCT_NAME')):

    for document in relations_documents_gen(relation_files):
        sentences = id_to_sent_dict(document)

        yield generate_positive(document, sentences, channels)
        # yield generate_negative(document, sentences, channels)


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
            f = relation.rel_from()
            t = relation.rel_to()
            f_sent_id = f.sentence_id()
            t_sent_id = t.sentence_id()

            f_element = get_relation_element(f, sentences)
            t_element = get_relation_element(t, sentences)
            f_indices = tuple(f_element.indices)
            t_indices = tuple(t_element.indices)

            relations[((f_sent_id, f_indices), (t_sent_id, t_indices))] = (
                relation, f_element.context, t_element.context)
            relations[((t_sent_id, t_indices), (f_sent_id, f_indices))] = (
                relation, t_element.context, f_element.context)

            for f_idx in f_indices:
                relidxs[(f_sent_id, f_idx)] = (
                f_indices, f_element.channel, f_element.ne)
            for t_idx in t_indices:
                relidxs[(t_sent_id, t_idx)] = (
                t_indices, t_element.channel, t_element.ne)

    for rel, rel_value in relations.items():
        relation, f_context, t_context = rel_value
        ((f_sent_id, f_indices), (t_sent_id, t_indices)) = rel

        f_nouns = get_nouns_idx(sentences[f_sent_id])
        t_nouns = get_nouns_idx(sentences[t_sent_id])

        if f_sent_id == t_sent_id:
            generator = permutations(f_nouns, 2)
        else:
            generator = product(f_nouns, t_nouns)

        for f_idx, t_idx in generator:
            try:
                _f_idxs, _f_channel_name, _f_ne = relidxs[(f_sent_id, f_idx)]
            except KeyError:
                _f_idxs = None
                _f_channel_name = ''
                _f_ne = 0.0
                pass

            try:
                _t_idxs, _t_channel_name, _t_ne = relidxs[(t_sent_id, t_idx)]
            except KeyError:
                _t_idxs = None
                _t_channel_name = ''
                _t_ne = 0.0
                pass

            if _t_idxs and _f_idxs and (
            (t_sent_id, _t_idxs), (f_sent_id, _f_idxs)) in relations:
                continue

            f_lemma = get_lemma(sentences[f_sent_id], f_idx)
            t_lemma = get_lemma(sentences[t_sent_id], t_idx)
            element_from = Relation.Element(f_sent_id, f_lemma,
                                            _f_channel_name, _f_ne, [f_idx],
                                            f_context)
            element_to = Relation.Element(t_sent_id, t_lemma, _t_channel_name,
                                          _t_ne, [t_idx], t_context)

            id_dir = get_document_dir(document)
            id_doc = get_document_file_name(document)
            rel = Relation(id_doc, element_from, element_to)
            lines.append(f'{id_dir}\tno_relation\t{rel}')
    return lines
