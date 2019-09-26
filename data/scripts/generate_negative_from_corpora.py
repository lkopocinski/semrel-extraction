#!/usr/bin/env python

import argparse
from itertools import permutations, product

from utils import corpora_files, load_document, id_to_sent_dict, \
    is_ner_relation, is_in_channel, get_relation_element, print_element, \
    get_nouns_idx, get_lemma

try:
    import argcomplete
except ImportError:
    argcomplete = None


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--source_directory', required=True,
                        help="A directory with corpora and relations files.")
    parser.add_argument('-c', '--channels', required=True,
                        help="A relation channels to be considered while generating set.")
    if argcomplete:
        argcomplete.autocomplete(parser)

    return parser.parse_args(argv)


def main(argv=None):
    args = get_args(argv)

    for corpora_file, relations_file in corpora_files(args.source_directory):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        relations = {}
        relidxs = {}
        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, args.channels):
                    f = relation.rel_from()
                    t = relation.rel_to()
                    f_sent_id = f.sentence_id()
                    t_sent_id = t.sentence_id()

                    f_lemma, f_idxs, f_context, f_channel_name = get_relation_element(f, sentences)
                    t_lemma, t_idxs, t_context, t_channel_name = get_relation_element(t, sentences)
                    f_idxs = tuple(f_idxs)
                    t_idxs = tuple(t_idxs)

                    relations[((f_sent_id, f_idxs), (t_sent_id, t_idxs))] = (relation, f_context, f_channel_name)
                    relations[((t_sent_id, t_idxs), (f_sent_id, f_idxs))] = (relation, t_context, t_channel_name)

                    for f_idx in f_idxs:
                        relidxs[(f_sent_id, f_idx)] = (f_idxs, f_channel_name)
                    for t_idx in t_idxs:
                        relidxs[(t_sent_id, t_idx)] = (t_idxs, t_channel_name)

        for rel, rel_value in relations.items():
            relation, f_context, f_channel_name = rel_value
            relation, t_context, t_channel_name = rel_value
            ((f_sent_id, f_idxs), (t_sent_id, t_idxs)) = rel

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
                print_element(f_lemma, t_lemma, _f_channel_name, _t_channel_name, f_idx, f_context, t_idx, t_context)


if __name__ == "__main__":
    main()
