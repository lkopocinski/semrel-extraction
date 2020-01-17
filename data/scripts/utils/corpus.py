from pathlib import Path
from typing import Iterator
from typing import List, Dict

import corpus2
from corpus_ccl import cclutils as ccl
from corpus_ccl import corpus_object_utils as cou
from corpus_ccl import token_utils as tou


class DocToken:

    def __init__(self, token: corpus2.Token):
        self._token = token
        self._lemma = self._get_lemma()

    @property
    def lemma(self):


    def is_ne(self, sentence):
        return tou.get_annotation(sentence, self._token, 'NE', default=0)

    def _get_lemma(self):
        try:
            return self._token.lexemes()[0].lemma_utf8()
        except IndexError:
            return ''


class DocSentence:

    def __init__(self, sentence: corpus2.Sentence):
        self._sentence = sentence

        self._tokens = self._get_tokens()
        self._orths = self._get_orths()
        self._lemmas = self._get_lemmas()

    @property
    def tokes(self):
        return self._tokens

    @property
    def orths(self):
        return self._orths


    def _get_tokens(self):
        return [token for token in self._sentence.tokens()]

    def _get_orths(self):
        return [token.orth_utf8() for token in self._tokens]

    def _get_lemmas(self):
        return

    def __str__(self):
        return ' '.join(self._orths)


class DocRelation:

    def __init__(self, relation: corpus2.Relation):
        self._relation = relation

        self._channel_from = self._relation.rel_from().channel_name()
        self._channel_to = self._relation.rel_to().channel_name()

    def is_ner(self):
        return self._relation.rel_set() == 'NER relation'

    @property
    def channels(self):
        return self._channel_from, self._channel_to

    def _get_member(self, relation_member, sentences: Dict[str, DocSentence]):
        sentence_id = relation_member.sentence_id()
        sentence = sentences[sentence_id]
        channel_name = relation_member.channel_name()
        annotation_number = relation_member.annotation_number()
        indices = get_annotation_indices(sentence, annotation_number, channel_name)

        if not indices:
            return None

        context = sentence.orths
        lemma = sentence.get_lemma(indices[0])
        any([sentence.tokes[idx] for idx in indices])
        ne = is_named_entity(sentence, indices)
        return Relation.Element(sentence_id, lemma, channel_name, ne, indices, context)

    def get_members(self, sentences: Dict[str, DocSentence]):
        member_from = self._get_member(self._relation.rel_from(), sentences)
        member_to = self._get_member(self._relation.rel_to(), sentences)
        return member_from, member_to


class Document:

    def __init__(self, document: corpus2.Document):
        self._document = document

        self._id = self._get_document_name()
        self._sentence_dict = self._id_to_sent_dict()
        self._relations = self._get_relations()

    @property
    def id(self) -> str:
        return self._id

    @property
    def relations(self) -> List[DocRelation]:
        return self._relations

    def get_sent(self, sentence_id: str) -> DocSentence:
        return self._sentence_dict[sentence_id]

    def _id_to_sent_dict(self) -> Dict[str, DocSentence]:
        return {sentence.id(): DocSentence(sentence)
                for par in self._document.paragraphs()
                for sentence in par.sentences()}

    def _get_document_name(self) -> str:
        ccl_path, __ = self._document.path().split(';')
        return Path(ccl_path).stem.split('.')[0]

    def _get_relations(self) -> List[DocRelation]:
        return [DocRelation(relation) for relation in self._document.relations()]

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._id == other._id
        return False


def documents_gen(corpus_files: List[Path]) -> Iterator[Document]:
    return (
        Document(ccl.read_ccl(str(path)))
        for path in corpus_files
        if path.is_file()
    )


def relations_documents_gen(relation_files: Iterator[Path]) -> Iterator[Document]:
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


def get_annotation_indices(sentences: corpus2.Sentence, ann_number: int, ann_channel: str):
    indices = []
    for index, token in enumerate(sentences.tokens()):
        number = tou.get_annotation(sentences, token, ann_channel, index, default=0)
        if number == ann_number:
            indices.append(index)
    return tuple(indices)


def get_context(sentence: corpus2.Sentence):
    return [token.orth_utf8() for token in sentence.tokens()]


def get_lemma(sentence: corpus2.Sentence, index: int):
    token = [token for token in sentence.tokens()][index]
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
