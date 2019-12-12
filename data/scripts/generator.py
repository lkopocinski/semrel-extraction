#!/usr/bin/env python3

from itertools import permutations, product
from pathlib import Path
from typing import List

from models import Relation
from utils.corpus import id_to_sent_dict, is_ner_relation, is_in_channel, get_relation_element, get_nouns_idx, \
    get_lemma, relations_documents_gen, get_document_ids


def generate_positive(relation_files: List[Path], channels):
    for document in relations_documents_gen(relation_files):
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    element_from = get_relation_element(relation.rel_from(), sentences)
                    element_to = get_relation_element(relation.rel_to(), sentences)

                    if element_from and element_to:
                        id_dir, id_doc = get_document_ids(document)
                        rel = Relation(id_doc, element_from, element_to)
                        yield f'{id_dir}\tin_relation\t{rel}'


def generate_negative(files, channels):
    for document in relations_documents_gen(files):
        sentences = id_to_sent_dict(document)

        relations = {}
        relidxs = {}
        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
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
                        relidxs[(f_sent_id, f_idx)] = (f_indices, f_element.channel, f_element.ne)
                    for t_idx in t_indices:
                        relidxs[(t_sent_id, t_idx)] = (t_indices, t_element.channel, t_element.ne)

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

                if _t_idxs and _f_idxs:
                    if ((t_sent_id, _t_idxs), (f_sent_id, _f_idxs)) in relations:
                        continue

                f_lemma = get_lemma(sentences[f_sent_id], f_idx)
                t_lemma = get_lemma(sentences[t_sent_id], t_idx)
                element_from = Relation.Element(f_sent_id, f_lemma, _f_channel_name, _f_ne, [f_idx], f_context)
                element_to = Relation.Element(t_sent_id, t_lemma, _t_channel_name, _t_ne, [t_idx], t_context)

                id_dir, id_doc = get_document_ids(document)
                rel = Relation(id_doc, element_from, element_to)
                yield f'{id_dir}\tno_relation\t{rel}'
