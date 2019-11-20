import abc

import numpy as np
import sent2vec
from gensim.models.fasttext import load_facebook_model
from wordfreq import zipf_frequency


class VectorizerFactory:
    @staticmethod
    def get_vectorizer(format, model_path):
        if not format:
            return None

        if format == 'sent2vec':
            return Sent2VecVectorizer(model_path)
        elif format == 'fasttext':
            return FastTextVectorizer(model_path)
        elif format == 'elmoconv':
            return ElmoConvolutionEngine()
        elif format == 'ner':
            return NamedEntityEngine()
        else:
            raise ValueError(format)


class Vectorizer(abc.ABC):

    @abc.abstractmethod
    def make_vectors(self, relation):
        pass


class Sent2VecVectorizer(Vectorizer):

    def __init__(self, model_path):
        self.s2v = sent2vec.Sent2vecModel()
        self.s2v.load_model(model_path, inference_mode=True)

    @staticmethod
    def mask_sentences(idx_f, ctx_f, idx_t, ctx_t):
        if ctx_f == ctx_t:
            ctx_f[int(idx_f)] = 'MASK'
            ctx_f[int(idx_t)] = 'MASK'
            sent_f = ' '.join(ctx_f)
            sent_t = ' '.join(ctx_f)
        else:
            ctx_f[int(idx_f)] = 'MASK'
            ctx_t[int(idx_t)] = 'MASK'
            sent_f = ' '.join(ctx_f)
            sent_t = ' '.join(ctx_t)

        return sent_f, sent_t

    def make_vectors(self, relation):
        sent_f, sent_t = self.mask_sentences(
            relation.source.index,
            relation.source.context,
            relation.dest.index,
            relation.dest.context
        )
        vf = self.s2v.embed_sentence(sent_f).flatten()
        vt = self.s2v.embed_sentence(sent_t).flatten()
        return vf, vt


class FastTextVectorizer(Vectorizer):

    def __init__(self, model_path):
        self.ft = load_facebook_model(model_path)

    def make_embedding(self, term):
        words = term.strip().split('_')
        embeddings = []
        weights = []

        for word in words:
            vec = self.ft[word]
            embeddings.append(vec)
            zipf_freq = zipf_frequency(word, 'pl')
            weights.append(1 / (zipf_freq if zipf_freq > 0 else 1))

        return np.average(embeddings, axis=0, weights=weights)

    def make_vectors(self, relation):
        lemma_f_vec = self.make_embedding(relation.source.lemma)
        lemma_t_vec = self.make_embedding(relation.dest.lemma)
        return lemma_f_vec, lemma_t_vec


class ElmoConvolutionEngine(Vectorizer):

    def make_vectors(self, relation):
        return relation.source.conv_vector, relation.dest.conv_vector


class NamedEntityEngine(Vectorizer):

    def make_vectors(self, relation):
        return [float(relation.source.ne)], [float(relation.dest.ne)]
