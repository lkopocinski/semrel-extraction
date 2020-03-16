from typing import List

from data.scripts.entities import Relation, Member
from data.scripts.utils.corpus import Document, DocSentence


def _make_token_key(id_domain: str, id_document: str, id_sentence: str, id_token: int) -> str:
    return f'{id_domain}\t{id_document}\t{id_sentence}\t{id_token}'


def make_token_key_member(id_domain: str, id_document: str, member: Member, id_token) -> str:
    return _make_token_key(id_domain, id_document, member.id_sentence, id_token)


def make_token_key_sentence(document: Document, sentence: DocSentence, id_token: int) -> str:
    return _make_token_key(document.directory, document.id, sentence.id, id_token)


def make_sentence_keys(document: Document, sentence: DocSentence) -> List[str]:
    return [make_token_key_sentence(document, sentence, id_token) for id_token, _ in enumerate(sentence.orths)]


def make_member_key(member: Member):
    return f'{member.id_sentence}' \
           f'\t{member.channel}' \
           f'\t{member.indices}' \
           f'\t{member.lemma}' \
           f'\t{member.is_named_entity}'


def make_relation_key(label: str, id_domain: str, relation: Relation) -> str:
    id_document, member_from, member_to = relation

    document_key = f'{label}\t{id_domain}\t{id_document}'
    member_from_key = make_member_key(member_from)
    member_to_key = make_member_key(member_to)

    return f'{document_key}\t{member_from_key}\t{member_to_key}'
