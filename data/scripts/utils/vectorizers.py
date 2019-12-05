import abc


import sent2vec
from allennlp.modules.elmo import Elmo, batch_to_ids

from model.models import Relation, Vector


class Vectorizer(abc.ABC):

    @abc.abstractmethod
    def embedd(self, relation: Relation):
        pass


class ElmoConvolutionVectorizer(Vectorizer):

    def __init__(self, options, weights):
        self.model = Elmo(options, weights, 1, dropout=0, scalar_mix_parameters=[1, -9e10, -9e10])

    def _make_vector(self, context, idx):
        character_ids = batch_to_ids([context])
        embeddings = self.model(character_ids)
        v = embeddings['elmo_representations'][0].data.numpy()
        value = v[:, idx, :].flatten()
        return Vector(value)

    def embedd(self, relation: Relation):
        v1 = self._make_vector(relation.source.context, relation.source.start_idx)
        v2 = self._make_vector(relation.dest.context, relation.dest.start_idx)
        return v1, v2


class Sent2VecVectorizer(Vectorizer):

    def __init__(self, model_path):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(model_path, inference_mode=True)

    @staticmethod
    def mask_sentences(index1, context1, index2, context2):
        if context1 == context2:
            context1[int(index1)] = 'MASK'
            context1[int(index2)] = 'MASK'
            sentence1 = ' '.join(context1)
            sentence2 = sentence1
        else:
            context1[int(index1)] = 'MASK'
            context2[int(index2)] = 'MASK'
            sentence1 = ' '.join(context1)
            sentence2 = ' '.join(context2)

        return sentence1, sentence2

    def _make_vector(self, sentence):
        value = self.model.embed_sentence(sentence).flatten()
        return Vector(value)

    def embedd(self, relation: Relation):
        sentence1, sentence2 = self.mask_sentences(
            relation.source.start_idx,
            relation.source.context,
            relation.dest.start_idx,
            relation.dest.context
        )

        v1 = self._make_vector(sentence1)
        v2 = self._make_vector(sentence2)
        return v1, v2


class NamedEntityVectorizer(Vectorizer):

    def _make_vector(self, value):
        value = [float(value)]
        return Vector(value)

    def embedd(self, relation):
        v1 = self._make_vector(relation.source.ne)
        v2 = self._make_vector(relation.dest.ne)
        return v1, v2
