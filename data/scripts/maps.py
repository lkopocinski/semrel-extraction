from pathlib import Path
from typing import Iterator, List

import torch

from data.scripts.utils.corpus import DocSentence, Document
from data.scripts.utils.vectorizers import Vectorizer


class MapMaker:

    def __init__(self, vectorizer: Vectorizer):
        self._vectorizer = vectorizer

    @staticmethod
    def make_keys(document: Document, sentence: DocSentence) -> List[str]:
        return [f'{document.directory}\t{document.id}\t{sentence.id}\t{id_token}'
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


class MapLoader:

    def __init__(self, keys_file: str, vectors_file: str):
        self._keys_path = Path(keys_file)
        self._vectors_file = Path(vectors_file)

    def _load_keys(self) -> dict:
        with self._keys_path.open('r', encoding='utf-8') as file:
            return {
                line.strip().split('\t'): index
                for index, line in enumerate(file)
            }

    def _load_vectors(self) -> torch.Tensor:
        return torch.load(self._vectors_file)

    def __call__(self) -> [dict, torch.Tensor]:
        keys = self._load_keys()
        vectors = self._load_vectors()
        return keys, vectors
