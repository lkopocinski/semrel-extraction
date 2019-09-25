import os
import zipfile
from allennlp.modules.elmo import Elmo, batch_to_ids

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


class ElmoEmb(object):

    def __init__(self, model_path):
        self._model = self._load_model(model_path)

    def _load_model(self, model_path):
        zipf = zipfile.ZipFile(model_path)
        zipf.extractall('./')
        zipf.close()

        model_dir = os.path.abspath('./')
        options_file = os.path.join(model_dir, 'emb-options.json')
        weights_file = os.path.join(model_dir, 'emb-weights.hdf5')
        return Elmo(options_file, weights_file, 2, dropout=0)

    def embedd(self, context, idx):
        character_ids = batch_to_ids([context])
        embeddings = self._model(character_ids)
        v = embeddings['elmo_representations'][1].data.numpy()
        return v[:, idx, :].flatten()
