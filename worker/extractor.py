import abc
from collections import deque
from itertools import product

from corpus_ccl import cclutils as ccl
from corpus_ccl import corpus_object_utils as cou
from corpus_ccl import token_utils as tou


class Parser(object):

    def __init__(self, extractor):
        self._slicer = SentenceWindow(window_size=3)
        self._extractor = extractor

    def __call__(self, document):
        for context in self._slicer.contextify(document):
            for first, second in self._extractor.extract(context, attr='NE'):
                yield first, second


class ContextType(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='default'):
        self._name = name

    @abc.abstractmethod
    def contextify(self, document):
        pass


class ExtractorType(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name='default', tagset='nkjp'):
        self._name = name
        self._tagset = ccl.get_tagset(tagset)

    def find_attribute_tokens(self, context, attr='default'):
        found = []
        for sentence in context:
            for ind, token in enumerate(sentence.tokens()):
                if tou.get_annotation(sentence, token, attr, ind, default=0):
                    ctx = [t.orth_utf8() for t in sentence.tokens()]
                    found.append((ind, ctx))
        return found

    def is_noun(self, token):
        return cou.get_coarse_pos(token, self._tagset) == 'noun'

    def is_ne(self, index, token, sentence):
        return tou.get_annotation(sentence, token, 'NE', index, default=0)

    def extract(self, context, attr='default'):
        analysed = self.find_attribute_tokens(context, attr)
        to_match = self._extract(context)
        return product(analysed, to_match)

    @abc.abstractmethod
    def _extract(self, context):
        pass


class NounExtractor(ExtractorType):

    def __init__(self, name='NounExtractor'):
        super(NounExtractor, self).__init__(name)

    def _extract(self, context):
        matched = []
        for sentence in context:
            for ind, token in enumerate(sentence.tokens()):
                if self.is_noun(token):
                    ctx = [t.orth_utf8() for t in sentence.tokens()]
                    matched.append((ind, ctx))
        return matched


class NERExtractor(ExtractorType):

    def __init__(self, name='NERExtractor'):
        super(NERExtractor, self).__init__(name)

    def _extract(self, context):
        matched = []
        for sentence in context:
            for ind, token in enumerate(sentence.tokens()):
                if self.is_ne(ind, token, sentence):
                    ctx = [t.orth_utf8() for t in sentence.tokens()]
                    matched.append((ind, ctx))
        return matched


class SentenceWindow(ContextType):

    def __init__(self, name='SentenceWindow', window_size=3, tagset='nkjp'):
        super(SentenceWindow, self).__init__(name)
        self._size = window_size

    def contextify(self, document):
        iterator = (sentence for paragraph in document.paragraphs()
                    for sentence in paragraph.sentences())

        window = deque(maxlen=self._size)

        for element in iterator:
            window.append(element)
            if len(window) == self._size:
                yield window
        if len(window) < self._size:
            yield window
