from itertools import permutations
from typing import Tuple, List

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
        orths = [orth
                 for indices, context in indices_context
                 for index, orth in enumerate(context)
                 if index in indices]

        vectors = [self._vectorizer.embed(context)[indices]
                   for indices, context in indices_context]
        vectors = torch.cat(vectors)

        orths_size = len(orths)
        orths_indices = range(orths_size)
        indices_pairs = [*permutations(orths_indices, 2)]
        idx_from, idx_to = zip(*indices_pairs)

        vec_from = vectors[[*idx_from]]
        vec_to = vectors[[*idx_to]]
        vector = torch.cat([vec_from, vec_to], 1)

        orths_pairs = [(orths[idx_f], orths[idx_t])
                       for idx_f, idx_t in indices_pairs]

        return orths_pairs, vector.to(self._device)

    def _predict(self, vectors: torch.Tensor):
        with torch.no_grad():
            predictions = self._net(vectors)
            return torch.argmax(predictions, 1)

    def predict(self, indices_context: List[Tuple]):
        orths, vectors = self._make_vectors(indices_context)
        predictions = self._predict(vectors)
        return orths, predictions.numpy()
