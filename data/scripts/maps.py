from typing import Iterator, List

import torch

from data.scripts.utils.corpus import DocSentence, Document
from data.scripts.utils.vectorizers import Vectorizer


class MapMaker:

    def __init__(self, vectorizer: Vectorizer):
        self._vectorizer = vectorizer

    @staticmethod
    def make_keys(document: Document, sentence: DocSentence) -> List[tuple]:
        return [(document.directory, document.id, sentence.id, id_token)
                for id_token, _ in enumerate(sentence.orths)]

    def make_sentence_map(self, document: Document, sentence: DocSentence) -> [List[tuple], torch.Tensor]:
        keys = self.make_keys(document, sentence)
        vectors = self._vectorizer.embed(sentence.orths)
        return keys, vectors

    def make_document_map(self, document: Document) -> [List[tuple], torch.Tensor]:
        keys = []
        vectors = []

        for sentence in document.sentences:
            sentence_keys, sentence_vectors = self.make_sentence_map(document, sentence)
            keys.extend(sentence_keys)
            vectors.extend(sentence_vectors)

        return keys, vectors

    def make_map(self, documents: Iterator[Document]) -> [List[tuple], torch.Tensor]:
        keys = []
        vectors = []

        for document in documents:
            document_keys, document_tensor = self.make_document_map(document)
            keys.extend(document_keys)
            vectors.extend(document_tensor)

        return keys, torch.stack(vectors)
