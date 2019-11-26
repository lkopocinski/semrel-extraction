import abc

import numpy as np
import sent2vec
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.models.KeyedVectors import load_word2vec_format
from gensim.models.fasttext import load_facebook_model
from wordfreq import zipf_frequency

from model.models import Relation, Vector


class VectorizerFactory:
    @staticmethod
    def get_vectorizer(format, model_path):
        if format == 'plain':
            return None

        if format == 'sent2vec':
            return Sent2VecVectorizer(model_path)
        elif format == 'fasttext':
            return FastTextVectorizer(model_path)
        elif format == 'elmoconv':
            return ElmoConvolutionVectorizer()
        elif format == 'ner':
            return NamedEntityVectorizer()
        elif format == 'retrofit':
            return RetrofitVectorizer(model_path)
        else:
            raise ValueError(format)


class Vectorizer(abc.ABC):

    @abc.abstractmethod
    def make_vectors(self, relation: Relation):
        pass


class ElmoVectorizer(Vectorizer):

    def __init__(self, options, weights):
        self.model = Elmo(options, weights, 1, dropout=0)

    def _make_vector(self, context, idx):
        character_ids = batch_to_ids([context])
        embeddings = self.model(character_ids)
        v = embeddings['elmo_representations'][0].data.numpy()
        value = v[:, idx, :].flatten()
        return Vector(value)

    def make_vectors(self, relation: Relation):
        v1 = self._make_vector(relation.source.context, relation.source.start_idx)
        v2 = self._make_vector(relation.dest.context, relation.dest.start_idx)
        return v1, v2


class ElmoConvolutionVectorizer(Vectorizer):

    def __init__(self, options, weights):
        self.model = Elmo(options, weights, 1, dropout=0, scalar_mix_parameters=[1, -9e10, -9e10])

    def _make_vector(self, context, idx):
        character_ids = batch_to_ids([context])
        embeddings = self.model(character_ids)
        v = embeddings['elmo_representations'][0].data.numpy()
        value = v[:, idx, :].flatten()
        return Vector(value)

    def make_vectors(self, relation: Relation):
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

    def make_vectors(self, relation: Relation):
        sentence1, sentence2 = self.mask_sentences(
            relation.source.start_idx,
            relation.source.context,
            relation.dest.start_idx,
            relation.dest.context
        )

        v1 = self._make_vector(sentence1)
        v2 = self._make_vector(sentence2)
        return v1, v2


class FastTextVectorizer(Vectorizer):

    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)

    def _make_vector(self, term):
        words = term.strip().split('_')
        embeddings = []
        weights = []

        for word in words:
            vec = self.model[word]
            embeddings.append(vec)
            zipf_freq = zipf_frequency(word, 'pl')
            weights.append(1 / (zipf_freq if zipf_freq > 0 else 1))

        value = np.average(embeddings, axis=0, weights=weights)
        return Vector(value)

    def make_vectors(self, relation: Relation):
        v1 = self._make_vector(relation.source.lemma)
        v2 = self._make_vector(relation.dest.lemma)
        return v1, v2


class RetrofitVectorizer(Vectorizer):

    def __init__(self, retrofitted_model_path, general_model_path):
        self.model_retrofit = load_word2vec_format(retrofitted_model_path)
        self.model_general = load_facebook_model(general_model_path)

    def _make_vector(self, term):
        value = []
        try:
            value = self.model_retrofit[term]
        except KeyError:
            print("Term not found in retrofit model: ", term)
            value = self.model_general[term]
        finally:
            return Vector(value)

    def make_vectors(self, relation: Relation):
        term1 = relation.source.context[relation.source.start_idx]
        term2 = relation.dest.context[relation.dest.start_idx]

        v1 = self._make_vector(term1)
        v2 = self._make_vector(term2)
        return v1, v2


class NamedEntityVectorizer(Vectorizer):

    def _make_vector(self, value):
        value = [float(value)]
        return Vector(value)

    def make_vectors(self, relation):
        v1 = self._make_vector(relation.source.ne)
        v2 = self._make_vector(relation.dest.ne)
        return v1, v2
