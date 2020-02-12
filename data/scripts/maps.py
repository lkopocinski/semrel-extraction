from typing import Iterator, List

import torch

import data.scripts.utils.vectorizers as vec
from data.scripts.utils.corpus import DocSentence, Document


class MapMaker:

    def __init__(self, vectorizer: vec.Vectorizer):
        self._vectorizer = vectorizer

    @staticmethod
    def make_keys(document: Document, sentence: DocSentence) -> List[tuple]:
        return [(document.directory, document.id, sentence.id, id_token)
                for id_token, _ in enumerate(sentence.orths)]

    def make_sentence_map(self, sentence: DocSentence, document: Document) -> [List[tuple], torch.Tensor]:
        keys = self.make_keys(document, sentence)
        vectors = self._vectorizer.embed(sentence.orths)

        return keys, vectors

    def make_map(self, documents: Iterator[Document]) -> [List[tuple], torch.Tensor]:
        keys = []
        vectors = []

        for document in documents:
            for sentence in document.sentences:
                _keys, _vectors = self.make_sentence_map(sentence, document)
                keys.extend(_keys)
                vectors.extend(_vectors)

        return keys, torch.stack(vectors)
