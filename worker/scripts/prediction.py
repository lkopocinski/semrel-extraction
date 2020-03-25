from itertools import permutations
from typing import Tuple, List

import numpy
import torch

from semrel.data.scripts.vectorizers import Vectorizer
from semrel.model.scripts import RelNet


class Predictor:

    def __init__(
            self,
            net_model: RelNet,
            vectorizer: Vectorizer,
            device: torch.device
    ):
        self._net = net_model
        self._vectorizer = vectorizer
        self._device = device

    def _make_vectors(
            self,
            indices_context: List[Tuple]
    ) -> Tuple[List[Tuple[str, str]], torch.Tensor]:
        orths = [
            orth
            for indices, context in indices_context
            for index, orth in enumerate(context)
            if index in indices
        ]

        vectors = [
            self._vectorizer.embed(context)[indices]
            for indices, context in indices_context
        ]
        vectors = torch.cat(vectors)

        orths_size = len(orths)
        orths_indices = range(orths_size)
        indices_pairs = [*permutations(orths_indices, r=2)]
        indices_from, indices_to = zip(*indices_pairs)

        vectors_from = vectors[[*indices_from]]
        vectors_to = vectors[[*indices_to]]
        vector = torch.cat([vectors_from, vectors_to], dim=1)

        orths_pairs = [
            (orths[idx_f], orths[idx_t])
            for idx_f, idx_t in indices_pairs
        ]

        return orths_pairs, vector.to(self._device)

    def _predict(self, vectors: torch.Tensor):
        with torch.no_grad():
            predictions = self._net(vectors)
            return torch.argmax(predictions, dim=1).cpu().numpy()

    def predict(
            self, indices_context: List[Tuple[List[int], List[str]]]
    ) -> [List[Tuple[str, str]], numpy.array]:
        orths, vectors = self._make_vectors(indices_context)
        predictions = self._predict(vectors)
        return orths, predictions
