from typing import NamedTuple, List, Tuple


class Member(NamedTuple):
    id_sentence: str
    lemma: str
    channel: str
    is_named_entity: bool
    indices: Tuple
    context: List[str]

    def __str__(self):
        return f'{self.id_sentence}\t{self.lemma}\t{self.channel}\t{self.is_named_entity}\t{self.indices}\t{self.context}'


class Relation(NamedTuple):
    id_document: str
    member_from: Member
    member_to: Member

    def __str__(self):
        return f'{self.id_document}\t{self.member_from}\t{self.member_to}'
