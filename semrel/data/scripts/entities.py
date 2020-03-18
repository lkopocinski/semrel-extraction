from typing import NamedTuple, List, Tuple


class Member(NamedTuple):
    id_sentence: str
    lemma: str
    channel: str
    is_named_entity: bool
    indices: Tuple
    context: List[str]

    def __str__(self):
        return f'{self.id_sentence}' \
               f'\t{self.lemma}' \
               f'\t{self.channel}' \
               f'\t{self.is_named_entity}' \
               f'\t{self.indices}' \
               f'\t{self.context}'


class Relation(NamedTuple):
    id_document: str
    member_from: Member
    member_to: Member

    def __str__(self):
        return f'{self.id_document}' \
               f'\t{self.member_from}' \
               f'\t{self.member_to}'
