#!/usr/bin/env python

from utils import find_token, get_context, corpora_files, load_document, id_to_sent_dict, is_ner_relation, is_in_channel, get_example
from constants import DIR, CHANNELS


def main():
    for corpora_file, relations_file in corpora_files(DIR):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, CHANNELS):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_idx, f_context = get_example(f, sentences)
                    t_idx, t_context = get_example(t, sentences)

                    if f_idx != -1 and t_idx != -1:
                        print('{}:{}\t{}:{}'.format(f_idx, f_context, t_idx, t_context))


if __name__ == "__main__":
    main()
