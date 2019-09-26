#!/usr/bin/env python

import argparse

from utils import corpora_files, load_document, id_to_sent_dict, \
    is_ner_relation, is_in_channel, get_relation_element, print_element, get_relation_element_multiword

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
    parser.add_argument('-m', '--multiword', type=bool, default=False,
                        help="Should generate in multiword mode or not")
    if argcomplete:
        argcomplete.autocomplete(parser)
    return parser.parse_args(argv)


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

                    if args.multiword:
                        f_lemma, f_idxs, f_context, f_channel_name = get_relation_element_multiword(f, sentences)
                        t_lemma, t_idxs, t_context, t_channel_name = get_relation_element_multiword(t, sentences)
                    else:
                        f_lemma, f_idxs, f_context, f_channel_name = get_relation_element(f, sentences)
                        t_lemma, t_idxs, t_context, t_channel_name = get_relation_element(t, sentences)

                    if f_idxs[0] != -1 and t_idxs[0] != -1:
                        print_element(
                            f_lemma, t_lemma,
                            f_channel_name, t_channel_name,
                            f_idxs, f_context,
                            t_idxs, t_context
                        )


if __name__ == "__main__":
    main()
