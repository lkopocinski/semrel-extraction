#!/usr/bin/env python
import random

from corpus_ccl import corpus_object_utils as cou

from utils import find_token, get_context, corpora_files, load_document, id_to_sent_dict, is_ner_relation, is_in_channel, get_example
from constants import DIR, CHANNELS


def is_noun(token):
    return 'subst' == cou.get_pos(token, 'nkjp', True)


def get_nouns(sent):
    return [idx for idx, token in enumerate(sent.tokens()) if is_noun(token)]


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

                    f_sent = sentences[f.sentence_id()]
                    f_nouns = get_nouns(f_sent)

                    t_sent = sentences[t.sentence_id()]
                    t_nouns = get_nouns(t_sent)

                    if f_context == t_context:
                        context = f_context
                        nouns = [idx for idx in f_nouns if idx not in (f_idx, t_idx)]

                        if not nouns:
                            continue

                        f_idx_noun = random.choice(nouns)
                        nouns.remove(f_idx_noun)

                        if not nouns:
                            continue

                        t_idx_noun = random.choice(nouns)
                        print('{}:{}\t{}:{}'.format(f_idx, context, t_idx_noun, context))
                        print('{}:{}\t{}:{}'.format(f_idx_noun, context, t_idx, context))
                    else:
                        f_nouns = [idx for idx in f_nouns if idx is not f_idx]
                        t_nouns = [idx for idx in t_nouns if idx is not t_idx]

                        if not f_nouns or not t_nouns:
                            continue

                        f_idx_noun = random.choice(f_nouns)
                        t_idx_noun = random.choice(t_nouns)

                        print('{}:{}\t{}:{}'.format(f_idx, f_context, t_idx_noun, t_context))
                        print('{}:{}\t{}:{}'.format(f_idx_noun, f_context, t_idx, t_context))


if __name__ == "__main__":
    main()
