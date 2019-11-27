from pathlib import Path
from typing import List

from corpus_ccl import cclutils as ccl
from corpus_ccl import corpus_object_utils as cou
from corpus_ccl import token_utils as tou

from data.scripts.model.models import Relation


def corpora_documents(relation_files: List[Path]):
    for rel_path in relation_files:
        ccl_path = Path(str(rel_path).replace('.rel', ''))
        if rel_path.is_file() and ccl_path.is_file():
            yield ccl.read_ccl_and_rel_ccl(str(ccl_path), str(rel_path))


def id_to_sent_dict(document):
    return {sentence.id(): sentence for par in document.paragraphs() for sentence in par.sentences()}


def is_ner_relation(relation):
    return relation.rel_set() == 'NER relation'


def is_in_channel(relation, channels):
    f_ch = relation.rel_from().channel_name()
    t_ch = relation.rel_to().channel_name()

    return f_ch in channels and t_ch in channels


def get_relation_element(rel, sentences):
    sent_id = rel.sentence_id()
    sent = sentences[sent_id]
    channel_name = rel.channel_name()
    indices = get_annotation_indices(sent, rel.annotation_number(), channel_name)

    if not indices:
        return None

    context = get_context(sent)
    lemma = get_lemma(sent, indices[0])
    ne = is_named_entity(sent, indices[0])

    return Relation.Element(sent_id, lemma, channel_name, indices, context, ne)


def get_annotation_indices(sent, ann_number, ann_channel):
    indices = []
    for index, token in enumerate(sent.tokens()):
        number = tou.get_annotation(sent, token, ann_channel, index, default=0)
        if number == ann_number:
            indices.append(index)
    return indices


def get_context(sent):
    return [token.orth_utf8() for token in sent.tokens()]


def get_lemma(sent, idx):
    return [token.lexemes()[0].lemma_utf8() for token in sent.tokens()][idx]


def get_document_name(document):
    ccl_path, rel_path = document.path().split(';')
    ccl_path = Path(ccl_path)
    return ccl_path.parent.stem, ccl_path.stem.split('.')[0]


def is_named_entity(sent, index):
    token = [token for token in sent.tokens()][index]
    ann = tou.get_annotation(sent, token, 'NE', index, default=0)
    return ann > 0


def get_nouns_idx(sent):
    return [idx for idx, token in enumerate(sent.tokens()) if is_noun(token)]


def is_noun(token):
    return 'subst' == cou.get_pos(token, 'nkjp', True)
