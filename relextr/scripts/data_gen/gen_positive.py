#!/usr/bin/env python

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

                    f_idx, f_context = get_example(f, sentences)
                    t_idx, t_context = get_example(t, sentences)

                    if f_idx != -1 and t_idx != -1:
                        print_element(f_idx, f_context, t_idx, t_context)


if __name__ == "__main__":
    main()
