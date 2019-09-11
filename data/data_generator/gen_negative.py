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

        relations = {}
        relidxs = {}
        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, args.channels):
                    f = relation.rel_from()
                    t = relation.rel_to()
                    s_f = f.sentence_id()
                    s_t = t.sentence_id()

                    _, f_ctx, f_idxs = get_example(f, sentences)
                    _, t_ctx, t_idxs = get_example(t, sentences)
                    f_idxs = tuple(f_idxs)
                    t_idxs = tuple(t_idxs)

                    relations[((s_f, f_idxs), (s_t, t_idxs))] = (relation, f_ctx)
                    relations[((s_t, t_idxs), (s_f, f_idxs))] = (relation, t_ctx)

                    for f_idx in f_idxs:
                        relidxs[(s_f, f_idx)] = f_idxs
                    for t_idx in t_idxs:
                        relidxs[(s_t, t_idx)] = t_idxs

        for ((s_f, f_idxs), (s_t, t_idxs)) in relations:
            relation, f_ctx = relations[((s_f, f_idxs), (s_t, t_idxs))]
            relation, t_ctx = relations[((s_f, f_idxs), (s_t, t_idxs))]

            f = relation.rel_from()
            t = relation.rel_to()

            if f_ctx == t_ctx:
                context = f_ctx
                nouns_idx = get_nouns_idx(sentences[s_f])
                nouns_idx = [idx for idx in nouns_idx if idx not in f_idxs + t_idxs]

                for idx_f, idx_t in permutations(nouns_idx, 2):
                    if are_close(idx_f, idx_t):
                        try:
                            _f_idxs = relidxs[(f.sentence_id(), idx_f)]
                        except KeyError:
                            _f_idxs = None

                        try:
                            _t_idxs = relidxs[(t.sentence_id(), idx_t)]
                        except KeyError:
                            _t_idxs = None

                        if _f_idxs and _t_idxs:
                            if ((f.sentence_id(), _f_idxs), (f.sentence_id(), _t_idxs)) in relations:
                                continue
                        print_element(idx_f, context, idx_t, context)
            else:
                f_nouns = get_nouns_idx(sentences[f.sentence_id()])
                t_nouns = get_nouns_idx(sentences[t.sentence_id()])

                f_nouns_idx = [idx for idx in f_nouns if idx not in f_idxs]
                t_nouns_idx = [idx for idx in t_nouns if idx not in t_idxs]

                for idx in f_nouns_idx:
                    try:
                        _f_idxs = relidxs[(f.sentence_id(), idx)]
                        if ((f.sentence_id(), _f_idxs), (t.sentence_id(), t_idxs)) in relations:
                            continue
                    except KeyError:
                        pass
                    print_element(idx, f_ctx, t_idxs[0], t_ctx)

                for idx in t_nouns_idx:
                    try:
                        _t_idxs = relidxs[(t.sentence_id(), idx)]
                        if ((t.sentence_id(), _t_idxs), (f.sentence_id(), f_idxs)) in relations:
                            continue
                    except KeyError:
                        pass
                    print_element(f_idxs[0], f_ctx, idx, t_ctx)

                for idx_f, idx_t in product(f_nouns_idx, t_nouns_idx):
                    print_element(idx_f, f_ctx, idx_t, t_ctx)


if __name__ == "__main__":
    main()
