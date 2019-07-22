import os

from corpus_ccl import cclutils as ccl
from corpus_ccl import token_utils as tou

from constants import RELATIONS_FILE_EXT


def corpora_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(RELATIONS_FILE_EXT):
            rel_file = os.path.join(directory, filename)
            corpus_file = os.path.join(directory, filename.replace('.rel', ''))

            if os.path.isfile(corpus_file):
                yield corpus_file, rel_file
        else:
            continue


def load_document(corpora_file, rel_file):
    return ccl.read_ccl_and_rel_ccl(corpora_file, rel_file)


def id_to_sent_dict(document):
    return {sentence.id(): sentence for par in document.paragraphs() for sentence in par.sentences()}


def find_token(sent, ann_number, ann_channel):
    for idx, token in enumerate(sent.tokens()):
        number = tou.get_annotation(sent, token, ann_channel)
        if number == ann_number:
            return idx, token
    return -1, None


def is_ner_relation(relation):
    return relation.rel_set() == 'NER relation'


def is_in_channel(relation, channels):
    f_ch = relation.rel_from().channel_name()
    t_ch = relation.rel_to().channel_name()

    return f_ch in channels and t_ch in channels

def get_context(sent):
    return [token.orth_utf8() for token in sent.tokens()]

def get_example(rel, sentences):
    sent = sentences[rel.sentence_id()]
    idx, token = find_token(sent, rel.annotation_number(), rel.channel_name())
    context = get_context(sent) if idx != -1 else None
    return idx, context