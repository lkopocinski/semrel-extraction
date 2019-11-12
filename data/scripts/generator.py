#!/usr/bin/env python3

from itertools import permutations, product

from model.models import Relation
from utils.corpus import corpora_files, load_document, id_to_sent_dict, \
    is_ner_relation, is_in_channel, get_relation_element, \
    get_nouns_idx, get_lemma, get_relation_element_multiword


def generate_positive(list_file, channels, multiword=False):
    for corpora_file, relations_file in corpora_files(list_file):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    if multiword:
                        f_element = get_relation_element_multiword(f, sentences)
                        t_element = get_relation_element_multiword(t, sentences)
                    else:
                        f_element = get_relation_element(f, sentences)
                        t_element = get_relation_element(t, sentences)

                    if f_element and t_element:
                        yield Relation(f_element, t_element)


def generate_negative(list_file, channels):
    for corpora_file, relations_file in corpora_files(list_file):
        document = load_document(corpora_file, relations_file)
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
                        relidxs[(f_sent_id, f_idx)] = (f_indices, f_element.channel)
                    for t_idx in t_indices:
                        relidxs[(t_sent_id, t_idx)] = (t_indices, t_element.channel)

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
                    _f_idxs, _f_channel_name = relidxs[(f_sent_id, f_idx)]
                except KeyError:
                    _f_idxs = None
                    _f_channel_name = ''
                    pass

                try:
                    _t_idxs, _t_channel_name = relidxs[(t_sent_id, t_idx)]
                except KeyError:
                    _t_idxs = None
                    _t_channel_name = ''
                    pass

                if _t_idxs and _f_idxs:
                    if ((t_sent_id, _t_idxs), (f_sent_id, _f_idxs)) in relations:
                        continue

                f_lemma = get_lemma(sentences[f_sent_id], f_idx)
                t_lemma = get_lemma(sentences[t_sent_id], t_idx)
                source = Relation.Element(f_lemma, _f_channel_name, [f_idx], f_context)
                target = Relation.Element(t_lemma, _t_channel_name, [t_idx], t_context)
                yield Relation(source, target)
