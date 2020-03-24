from itertools import chain
from pathlib import Path
from typing import Iterator, List, NamedTuple, Dict

import torch

from semrel.data.scripts.utils.keys import make_sentence_keys
from semrel.data.scripts.corpus import DocSentence, Document
from semrel.data.scripts.vectorizers import Vectorizer


class VectorsMap(NamedTuple):
    keys: Dict
    vectors: torch.Tensor


class MapLoader:

    def __init__(self, keys_file: str, vectors_file: str):
        self._keys_path = Path(keys_file)
        self._vectors_file = Path(vectors_file)

    def _load_keys(self) -> Dict:
        with self._keys_path.open('r', encoding='utf-8') as file:
            return {line.strip(): index for index, line in enumerate(file)}

    def _load_vectors(self) -> torch.Tensor:
        return torch.load(str(self._vectors_file))

    def __call__(self) -> VectorsMap:
        keys = self._load_keys()
        vectors = self._load_vectors()
        return VectorsMap(keys, vectors)


class MapMaker:

    def __init__(self, vectorizer: Vectorizer):
        self._vectorizer = vectorizer

    def make_sentence_map(
            self, document: Document, sentence: DocSentence
    ) -> [List[str], torch.Tensor]:
        keys = make_sentence_keys(document, sentence)
        vectors = self._vectorizer.embed(sentence.orths)
        return keys, vectors

    def make_document_map(
            self, document: Document
    ) -> [List[str], torch.Tensor]:
        sentences_maps = [self.make_sentence_map(document, sentence)
                          for sentence in document.sentences]
        keys, vectors = zip(*sentences_maps)

        keys = list(chain(*keys))
        vectors = torch.cat(vectors)
        return keys, vectors

    def make_map(
            self, documents: Iterator[Document]
    ) -> [List[str], torch.Tensor]:
        document_maps = [self.make_document_map(document)
                         for document in documents]
        keys, vectors = zip(*document_maps)

        keys = list(chain(*keys))
        vectors = torch.cat(vectors)
        return keys, vectors
