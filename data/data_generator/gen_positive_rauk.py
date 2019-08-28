#!/usr/bin/env python

from utils import find_token, get_context, corpora_files, load_document, id_to_sent_dict, is_ner_relation, is_in_channel, get_example_
from constants import DIR, CHANNELS

def get_example_(rel, sentences):
    sent = sentences[rel.sentence_id()]
    idx, token = find_token(sent, rel.annotation_number(), rel.channel_name())
    if idx == -1:
        return -1, None
    try:
        lemma = tou.get_attributes(token)['BRAND_NAME:Forma podstawowa']
        if lemma == '':
            raise
    except:
        try:
            lemma = tou.get_attributes(token)['BRAND_NAME:lemma']
            if lemma == '':
                raise
            
        except:
            try:
                lemma = tou.get_attributes(token)['PRODUCT_NAME:Forma podstawowa']
                if lemma == '':
                    raise
            except:
                try:
                    lemma = tou.get_attributes(token)['PRODUCT_NAME:lemma']
                    if lemma == '':
                        raise
                except:
                    idx = -1
                    lemma = None
    return idx, lemma

def main():
    for corpora_file, relations_file in corpora_files(DIR):
        document = load_document(corpora_file, relations_file)
        sentences = id_to_sent_dict(document)

        for relation in document.relations():
            if is_ner_relation(relation):
                if is_in_channel(relation, CHANNELS):
                    f = relation.rel_from()
                    t = relation.rel_to()

                    f_idx, f_lemma = get_example_(f, sentences)
                    t_idx, t_lemma = get_example_(t, sentences)

#                    print(corpora_file)
                    if f_idx != -1 and t_idx != -1:
                        if f.channel_name() == 'BRAND_NAME' and t.channel_name() == 'PRODUCT_NAME':
                            print('{}\t{}\t{}'.format(f_lemma,
                                                 'product',
                                                 t_lemma))
#                            print('{}'.format(f_context))
#                            print('{}'.format(t_context))
                    elif f.channel_name() == 'PRODUCT_NAME' and t.channel_name() == 'BRAND_NAME':
                            print('{}\t{}\t{}'.format(t_lemma,
                                                 'product',
                                                 f_lemma))
#                            print('{}'.format(t_context))
#                            print('{}'.format(f_context))



if __name__ == "__main__":
    main()
