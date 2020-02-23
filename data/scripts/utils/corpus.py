from itertools import chain
from pathlib import Path
from typing import Iterator
from typing import List, Dict, Tuple

import corpus2
from corpus_ccl import cclutils as ccl
from corpus_ccl import corpus_object_utils as cou
from corpus_ccl import token_utils as tou

from data.scripts.entities import Member


class DocToken:

    def __init__(self, token: corpus2.Token):
        self._token = token

        self._lemma = self._get_lemma()
        self._orth = self._get_orth()
        self._pos = self._get_pos()

    @property
    def lemma(self) -> str:
        return self._lemma

    @property
    def orth(self) -> str:
        return self._orth

    @property
    def is_noun(self) -> bool:
        return 'subst' == self._pos

    def _get_lemma(self) -> str:
        try:
            return self._token.lexemes()[0].lemma_utf8()
        except IndexError:
            return ''

    def _get_orth(self) -> str:
        return self._token.orth_utf8()

    def _get_pos(self) -> str:
        try:
            return cou.get_pos(self._token, 'nkjp', True)
        except IndexError:
            return ''


class DocSentence:

    def __init__(self, sentence: corpus2.Sentence):
        self._sentence = sentence

        self._id = self._sentence.id()
        self._tokens = self._get_tokens()
        self._orths = self._get_orths()
        self._lemmas = self._get_lemmas()
        self._named_entities_indices = self._get_named_entities_indices()
        self._noun_indices = self._get_noun_indices()

    @property
    def id(self) -> str:
        return self._id

    @property
    def tokens(self) -> List[DocToken]:
        return self._tokens

    @property
    def orths(self) -> List[str]:
        return self._orths

    @property
    def lemmas(self) -> List[str]:
        return self._lemmas

    @property
    def named_entities_indices(self) -> List[int]:
        return self._named_entities_indices

    @property
    def noun_indices(self) -> List[int]:
        return self._noun_indices

    def _get_tokens(self) -> List[DocToken]:
        return [DocToken(token) for token in self._sentence.tokens()]

    def _get_orths(self) -> List[str]:
        return [token.orth for token in self._tokens]

    def _get_lemmas(self) -> List[str]:
        return [token.lemma for token in self._tokens]

    def _get_named_entities_indices(self) -> List[int]:
        return [index
                for index, token in enumerate(self._tokens)
                if tou.get_annotation(self._sentence, token._token, 'NAMED_ENTITY', default=False)]

    def _get_noun_indices(self) -> List[int]:
        return [index for index, token in enumerate(self._tokens) if token.is_noun]

    def __str__(self):
        return ' '.join(self._orths)


class DocRelation:

    def __init__(self, relation: corpus2.Relation, sentences: Dict[str, DocSentence]):
        self._relation = relation

        self._is_ner = self._is_ner_relation()
        self._channel_from = self._relation.rel_from().channel_name()
        self._channel_to = self._relation.rel_to().channel_name()

        self._sentence_from = sentences[self._relation.rel_from().sentence_id()]
        self._sentence_to = sentences[self._relation.rel_to().sentence_id()]

    @property
    def is_ner(self) -> bool:
        return self._is_ner

    @property
    def channels(self) -> [str, str]:
        return self._channel_from, self._channel_to

    def _is_ner_relation(self) -> bool:
        return self._relation.rel_set() == 'NER relation'

    def _get_member(self, relation_member: corpus2.DirectionPoint, sentence: DocSentence):
        sentence_id = relation_member.sentence_id()
        channel_name = relation_member.channel_name()
        indices = self._get_annotation_indices(relation_member, sentence)

        if not indices:
            return None

        context = sentence.orths
        lemma = sentence.lemmas[indices[0]]
        named_entity = all(index in sentence.named_entities_indices for index in indices)
        return Member(sentence_id, lemma, channel_name, named_entity, indices, context)

    @staticmethod
    def _get_annotation_indices(relation_member: corpus2.DirectionPoint, sentence: DocSentence) -> Tuple[int]:
        indices = []

        for index, token in enumerate(sentence.tokens):
            annotation_number = tou.get_annotation(
                sentence._sentence, token._token, relation_member.channel_name(), index, default=0
            )
            if annotation_number == relation_member.annotation_number():
                indices.append(index)

        return tuple(indices)

    def get_members(self) -> [Member, Member]:
        member_from = self._get_member(self._relation.rel_from(), self._sentence_from)
        member_to = self._get_member(self._relation.rel_to(), self._sentence_to)
        return member_from, member_to


class Document:

    def __init__(self, document: corpus2.Document):
        self._document = document

        self._id = self._get_document_name()
        self._directory = self._get_document_directory()
        self._sentences = self._get_sentences()
        self._sentence_dict = self._id_to_sentence_dict()
        self._relations = self._get_relations()

    @property
    def id(self) -> str:
        return self._id

    @property
    def directory(self) -> str:
        return self._directory

    @property
    def sentences(self) -> List[DocSentence]:
        return self._sentences

    @property
    def relations(self) -> List[DocRelation]:
        return self._relations

    def get_sentence(self, sentence_id: str) -> DocSentence:
        return self._sentence_dict.get(sentence_id, None)

    def _id_to_sentence_dict(self) -> Dict[str, DocSentence]:
        return {sentence.id(): DocSentence(sentence)
                for paragraph in self._document.paragraphs()
                for sentence in paragraph.sentences()}

    def _get_document_name(self) -> str:
        ccl_path, __ = self._document.path().split(';')
        return Path(ccl_path).stem.split('.')[0]

    def _get_document_directory(self) -> str:
        ccl_path, __ = self._document.path().split(';')
        return Path(ccl_path).parent.stem

    def _get_sentences(self) -> List[DocSentence]:
        return [DocSentence(sentence)
                for paragraph in self._document.paragraphs()
                for sentence in paragraph.sentences()]

    def _get_relations(self) -> List[DocRelation]:
        return [DocRelation(relation, self._sentence_dict) for relation in self._document.relations()]

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._id == other._id
        return False


def from_index_documents_gen(relations_files_index: Path) -> Iterator[Document]:
    with relations_files_index.open('r', encoding='utf-8') as file:
        relation_files = [Path(line.strip()) for line in file]
        return documents_gen(relation_files)


def documents_gen(relations_files: Iterator[Path]) -> Iterator[Document]:
    for rel_path in relations_files:
        ccl_path = Path(str(rel_path).replace('rel', 'ner'))
        if rel_path.is_file() and ccl_path.is_file():
            ccl_document = ccl.read_ccl_and_rel_ccl(
                ccl_file=str(ccl_path), rel_ccl_file=str(rel_path)
            )
            yield Document(ccl_document)


def relations_files_paths(corpora_path: str, directories: List) -> Iterator[Path]:
    return list(chain.from_iterable(
        dir_path.glob('*.rel.xml')
        for dir_path in Path(corpora_path).iterdir()
        if dir_path.stem in directories))
