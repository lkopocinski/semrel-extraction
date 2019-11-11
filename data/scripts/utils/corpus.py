import os
from models import Relation

from corpus_ccl import cclutils as ccl
from corpus_ccl import corpus_object_utils as cou
from corpus_ccl import token_utils as tou


def corpora_files(paths_file):
    with open(paths_file, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            filepath = line.strip()
            if filepath.endswith('rel.xml'):
                rel_file = filepath
                corpus_file = filepath.replace('.rel', '')
                if os.path.isfile(corpus_file):
                    yield corpus_file, rel_file
            else:
                continue


def load_document(corpora_file, rel_file):
    return ccl.read_ccl_and_rel_ccl(corpora_file, rel_file)


def id_to_sent_dict(document):
    return {sentence.id(): sentence for par in document.paragraphs() for sentence in par.sentences()}


def is_ner_relation(relation):
    return relation.rel_set() == 'NER relation'


def is_in_channel(relation, channels):
    f_ch = relation.rel_from().channel_name()
    t_ch = relation.rel_to().channel_name()

    return f_ch in channels and t_ch in channels


def get_relation_element(rel, sentences):
    sent = sentences[rel.sentence_id()]
    channel_name = rel.channel_name()
    indices = find_token_indices(sent, rel.annotation_number(), channel_name)

    if not indices:
        return -1, None

    context = get_context(sent)
    lemma = get_lemma(sent, indices[0])
    return Relation.Element(lemma, channel_name, indices, context)


def find_token_indices(sent, ann_number, ann_channel):
    idxs = []
    for idx, token in enumerate(sent.tokens()):
        number = tou.get_annotation(sent, token, ann_channel)
        if number == ann_number:
            idxs.append(idx)
    return idxs


def get_context(sent):
    return [token.orth_utf8() for token in sent.tokens()]


def get_lemma(sent, idx):
    return [token.lexemes()[0].lemma_utf8() for token in sent.tokens()][idx]


def get_nouns_idx(sent):
    return [idx for idx, token in enumerate(sent.tokens()) if is_noun(token)]


def is_noun(token):
    return 'subst' == cou.get_pos(token, 'nkjp', True)


def get_relation_element_multiword(rel, sentences):
    sent = sentences[rel.sentence_id()]
    channel_name = rel.channel_name()
    indices = find_token_indices(sent, rel.annotation_number(), channel_name)

    if not indices:
        return -1, None

    begin = indices[0]
    end = indices[-1]

    context = get_context(sent)
    phrase = ' '.join(context[begin:end + 1])
    context[begin:end + 1] = [phrase]

    lemma = get_multiword_lemma(sent, indices[0])
    return Relation.Element(lemma, channel_name, [begin], context)


def get_multiword_lemma(sent, idx):
    token = [token for token in sent.tokens()][idx]
    try:
        lemma = tou.get_attributes(token)['BRAND_NAME:lemma']
        if lemma == '':
            raise Exception
    except:
        try:
            lemma = tou.get_attributes(token)['BRAND_NAME:Forma podstawowa']
            if lemma == '':
                raise Exception
        except:
            try:
                lemma = tou.get_attributes(token)['PRODUCT_NAME:Forma podstawowa']
                if lemma == '':
                    raise Exception
            except:
                try:
                    lemma = tou.get_attributes(token)['PRODUCT_NAME:lemma']
                    if lemma == '':
                        raise Exception
                except:
                    lemma = get_lemma(sent, idx)
    return lemma
