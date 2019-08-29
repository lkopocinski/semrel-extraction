#!/usr/bin/env python

from itertools import permutations, product

from corpus_ccl import corpus_object_utils as cou

from utils import corpora_files, load_document, id_to_sent_dict, \
    is_ner_relation, is_in_channel, get_example, print_element

import argparse

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


def is_noun(token):
    return 'subst' == cou.get_pos(token, 'nkjp', True)


def get_nouns_idx(sent):
    return [idx for idx, token in enumerate(sent.tokens()) if is_noun(token)]


def are_close(idx_f, idx_t):
    return abs(idx_f - idx_t) <= 3


def main(argv=None):
    args = get_args(argv)

    for corpora_file, relations_file in corpora_files(args.source_directory):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, args.channels):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_idx, f_context, f_idxs = get_example(f, sentences)
                    t_idx, t_context, t_idxs = get_example(t, sentences)

                    if f_context == t_context:
                        context = f_context
                        nouns_idx = get_nouns_idx(sentences[f.sentence_id()])
                        nouns_idx = [idx for idx in nouns_idx if idx not in f_idxs]

                        for idx in nouns_idx:
                            if are_close(f_idx, idx):
                                print_element(f_idx, context, idx, context)
                            if are_close(t_idx, idx):
                                print_element(idx, context, t_idx, context)

                        for idx_f, idx_t in permutations(nouns_idx, 2):
                            if are_close(idx_f, idx_t):
                                print_element(idx_f, context, idx_t, context)
                    else:
                        f_nouns = get_nouns_idx(sentences[f.sentence_id()])
                        t_nouns = get_nouns_idx(sentences[t.sentence_id()])

                        f_nouns_idx = [idx for idx in f_nouns if idx not in f_idxs]
                        t_nouns_idx = [idx for idx in t_nouns if idx not in t_idxs]

                        for idx in f_nouns_idx:
                            print_element(idx, f_context, t_idx, t_context)

                        for idx in t_nouns_idx:
                            print_element(f_idx, f_context, idx, t_context)

                        for idx_f, idx_t in product(f_nouns_idx, t_nouns_idx):
                            print_element(idx_f, f_context, idx_t, t_context)


if __name__ == "__main__":
    main()
