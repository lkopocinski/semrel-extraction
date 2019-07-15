from allennlp.modules.elmo import Elmo, batch_to_ids


class ElmoEmb(object):

    def __init__(self, options_file, weights_file):
        self._model = Elmo(options_file, weights_file, 2, dropout=0)

    def embedd(self, context, idx):
        character_ids = batch_to_ids([context])
        embeddings = self._model(character_ids)
        v = embeddings['elmo_representations'][1].data.numpy()
        return v[:, idx, :].flatten()
