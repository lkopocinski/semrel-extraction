import abc
from abc import ABC
from collections import deque
from typing import Deque, Tuple, Callable, List

from semrel.data.scripts.corpus import Document, DocSentence


class Slicer(ABC):

    def __init__(self, name: str = 'default'):
        self._name = name

    @abc.abstractmethod
    def contextify(self, document: Document):
        pass


class SentenceWindow(Slicer):

    def __init__(self, name: str = 'SentenceWindow', window_size: int = 3):
        super(SentenceWindow, self).__init__(name)
        self._size = window_size

    def contextify(self, document: Document) -> Deque:
        window = deque(maxlen=self._size)

        for sentence in document.sentences:
            window.append(sentence)
            if len(window) == self._size:
                yield window

        if len(window) < self._size:
            yield window


class Parser:

    def __init__(self, slicer: Slicer, extractor: Callable):
        self._slicer = slicer
        self._extractor = extractor

    def __call__(self, document: Document) -> List[Tuple]:
        for context in self._slicer.contextify(document):
            yield self._extractor(context)


def find_nouns(context: Deque[DocSentence]) -> List[Tuple]:
    return [
        (sentence.noun_indices, sentence.orths)
        for sentence in context
    ]


def find_named_entities(context: Deque[DocSentence]) -> List[Tuple]:
    return [
        (sentence.named_entities_indices, sentence.orths)
        for sentence in context
    ]
