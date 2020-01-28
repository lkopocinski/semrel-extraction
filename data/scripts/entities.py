from typing import NamedTuple, List


class Element(NamedTuple):
    sent_id: str
    lemma: str
    channel: str
    ne: bool
    indices: tuple
    context: List[str]

    @property
    def start_idx(self):
        return self.indices[0]

    def __str__(self):
        return f'{self.sent_id}\t{self.lemma}\t{self.channel}\t{self.ne}\t{self.indices}\t{self.context}'


class Relation(NamedTuple):
    document_id: str
    member_from: Element
    member_to: Element

    def __str__(self):
        return f'{self.document_id}\t{self.member_from}\t{self.member_to}'
