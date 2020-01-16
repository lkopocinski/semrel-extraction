from pathlib import Path
from typing import Generator, Iterator
from typing import List

import corpus2
from corpus_ccl import cclutils as ccl
from corpus_ccl import corpus_object_utils as cou
from corpus_ccl import token_utils as tou

from data.scripts.models import Relation


class Document:

    def __init__(self, document: corpus2.Document):
        self._document = document
        self._id = self._get_document_name()

    @property
    def id(self):
        return self._id

    def _get_document_name(self):
        ccl_path, __ = self._document.path().split(';')
        return Path(ccl_path).stem.split('.')[0]

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._id == other._id
        return False


def documents_gen(corpus_files: List[Path]) -> Generator[
    corpus2.Document, None, None]:
    return (
        ccl.read_ccl(path)
        for path in corpus_files
        if Path(path).is_file()
    )


def relations_documents_gen(relation_files: Iterator[Path]):
    for rel_path in relation_files:
        ccl_path = Path(str(rel_path).replace('.rel', ''))
        if rel_path.is_file() and ccl_path.is_file():
            yield Document(ccl.read_ccl_and_rel_ccl(str(ccl_path), str(rel_path)))


def id_to_sent_dict(document):
    return {sentence.id(): sentence
            for par in document.paragraphs()
            for sentence in par.sentences()}


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
    indices = get_annotation_indices(sent, rel.annotation_number(),
                                     channel_name)

    if not indices:
        return None

    context = get_context(sent)
    lemma = get_lemma(sent, indices[0])
    ne = is_named_entity(sent, indices)
    return Relation.Element(sent_id, lemma, channel_name, ne, indices, context)


def get_annotation_indices(sent, ann_number, ann_channel):
    indices = []
    for index, token in enumerate(sent.tokens()):
        number = tou.get_annotation(sent, token, ann_channel, index, default=0)
        if number == ann_number:
            indices.append(index)
    return tuple(indices)


def get_context(sent):
    return [token.orth_utf8() for token in sent.tokens()]


def get_lemma(sent, index):
    token = [token for token in sent.tokens()][index]
    try:
        return token.lexemes()[0].lemma_utf8()
    except IndexError:
        return ''


def get_document_dir(document):
    ccl_path, __ = document.path().split(';')
    return Path(ccl_path).parent.stem


def get_document_file_name(document):
    ccl_path, __ = document.path().split(';')
    return Path(ccl_path).stem.split('.')[0]


def get_sentence_id(sentence):
    return sentence.id()


def is_named_entity(sent, indices):
    annotations = [tou.get_annotation(sent, token, 'NE', index, default=0) for
                   index, token in enumerate(sent.tokens())
                   if index in indices]
    return all(annotations)


def get_nouns_idx(sent):
    return [idx for idx, token in enumerate(sent.tokens()) if is_noun(token)]


def is_noun(token):
    try:
        return 'subst' == cou.get_pos(token, 'nkjp', True)
    except IndexError:
        return False
