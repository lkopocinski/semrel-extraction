import abc
from collections import deque

from corpus_ccl import cclutils as ccl


class Extractor(object):

    def __init__(self):
        self._slicer = SentenceWindow(window_size=2)

    def __call__(self, context):
        pass

    def _doc2ctx(self, document, attr='mwe_base'):
        for context in self._slicer.contextify(document):
            yield self._slicer.extract(context, attr)


class ContextType(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, name='default'):
        self._name = name

    @abc.abstractmethod
    def extract(self, context):
        pass

    @abc.abstractmethod
    def contextify(self, document):
        pass


class SentenceWindow(ContextType):

    def __init__(self, name='SentenceWindow', window_size=3):
        self._name = name
        self._size = window_size

    def contextify(self, document):
        iterator = (sentence for paragraph in document.paragraphs()
                for sentence in paragraph.sentences())

        window = deque(maxlen=self._size)

        for ind, element in iterator:
            window.put(element)
            if len(window) == self._size:
                yield window
        if len(window) < self._size:
            yield window


e = Extractor()
d = ccl.read_ccl('./test.xml')
for ctx in e._doc2ctx(d):
    pass
