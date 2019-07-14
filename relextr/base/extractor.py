import abc
from itertools import product
from collections import deque

from corpus_ccl import cclutils as ccl
from corpus_ccl import corpus_object_utils as cou


class Parser(object):

    def __init__(self):
        self._slicer = SentenceWindow(window_size=3)
        self._extractor = NounExtractor()

    def __call__(self, context):
        pass

    def _doc2ctx(self, document):
        for context in self._slicer.contextify(document):
            yield self._extractor.extract(context)


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

    def tokens(self, context):
        return (token for sentence in context for token in sentence.tokens())

    def find_attribute_tokens(self, context, attr='default'):
        found = set()
        for token in self.tokens(context):
            if not token.has_metadata():
                continue
            metadata = token.get_metadata()
            if not metadata.has_attribute(attr):
                continue
            found.add(metadata.get_attribute(attr))
        return found

    def is_noun(self, token):
        return cou.get_coarse_pos(token, self._tagset) == 'noun'

    @abc.abstractmethod
    def extract(self, context):
        pass


class NounExtractor(ExtractorType):

    def __init__(self, name='NounExtractor'):
        super(NounExtractor, self).__init__(name)

    def extract(self, context, attr='BRAND'):
        analysed = self.find_attribute_tokens(context, attr)
        matched = set()

        for token in self.tokens(context):
            if not token.has_metadata():
                continue
            metadata = token.get_metadata()
            if not metadata.has_attribute(attr):
                continue
            if self.is_noun(token):
                matched.add(cou.get_lexeme_lemma(token))
        return product(analysed, matched)


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


e = Parser()
d = ccl.read_ccl('./test.xml')
for ctx in e._doc2ctx(d):
    pass
