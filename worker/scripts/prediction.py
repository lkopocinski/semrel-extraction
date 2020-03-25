from itertools import permutations
from typing import Tuple, List

import torch

from semrel.data.scripts.vectorizers import ElmoVectorizer, FastTextVectorizer
from semrel.model.scripts import RelNet


class Predictor:

    def __init__(
            self,
            net_model: RelNet,
            elmo: ElmoVectorizer,
            fasttext: FastTextVectorizer,
            device: torch.device
    ):
        self._net = net_model
        self._elmo = elmo
        self._fasttext = fasttext
        self._device = device

    def _make_vectors(self, indices_context: List[Tuple]):
        orths = []
        vectors = []

        for indices, context in indices_context:
            orths_pairs = [
                orth
                for index, orth in enumerate(context)
                if index in indices
            ]
            _vectors_elmo = self._elmo.embed(context)
            _vectors_fasttext = self._fasttext.embed(context)

            _vectors_elmo = _vectors_elmo[indices]
            _vectors_fasttext = _vectors_fasttext[indices]

            orths.extend(orths_pairs)
            vectors.append((_vectors_elmo, _vectors_fasttext))

        vectors_elmo, vectors_fasttext = zip(*vectors)
        vectors_elmo = torch.cat(vectors_elmo)
        vectors_fasttext = torch.cat(vectors_fasttext)

        size = len(orths)

        idx_from, idx_to = zip(*list(permutations(range(size), 2)))

        elmo_from = vectors_elmo[[*idx_from]]
        elmo_to = vectors_elmo[[*idx_to]]

        fasttext_from = vectors_fasttext[[*idx_from]]
        fasttext_to = vectors_fasttext[[*idx_to]]

        elmo_vectors = torch.cat([elmo_from, elmo_to], 1)
        fasttext_vectors = torch.cat([fasttext_from, fasttext_to], 1)

        vector = torch.cat([elmo_vectors, fasttext_vectors], 1)

        orths_pairs = [(orths[idx_f], orths[idx_t])
                  for idx_f, idx_t in zip(idx_from, idx_to)]
        return orths_pairs, vector.to(self._device)

    def _predict(self, vectors: torch.Tensor):
        with torch.no_grad():
            predictions = self._net(vectors)
            return torch.argmax(predictions)

    def predict(self, indices_context: List[Tuple]):
        orths, vectors = self._make_vectors(indices_context)
        predictions = self._predict(vectors)
        return zip(orths, predictions)
