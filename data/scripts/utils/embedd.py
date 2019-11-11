import sys

import numpy
from allennlp.modules.elmo import Elmo, batch_to_ids
from model.models import Vector
numpy.set_printoptions(threshold=sys.maxsize)


class ElmoEmb(object):

    def __init__(self, options_file, weights_file, layer: str = 'average'):
        self._model = self._get_model(options_file, weights_file, layer)

    def _get_model(self, options_file, weights_file, layer):
        if layer == 'average':
            return Elmo(options_file, weights_file, 1, dropout=0)
        elif layer == 'convolution':
            return Elmo(options_file, weights_file, 1, dropout=0, scalar_mix_parameters=[1, -9e10, -9e10])

    def embedd(self, element):
        character_ids = batch_to_ids([element.context])
        embeddings = self._model(character_ids)
        v = embeddings['elmo_representations'][0].data.numpy()
        value = v[:, element.index, :].flatten()
        return Vector(value)
