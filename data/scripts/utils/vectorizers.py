import abc
from typing import List

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model


class Vectorizer(abc.ABC):

    @abc.abstractmethod
    def embed(self, context: List[str]):
        pass


class ElmoVectorizer(Vectorizer):

    def __init__(self, options, weights):
        self.model = Elmo(options, weights, 1, dropout=0)

    def embed(self, context):
        character_ids = batch_to_ids([context])
        embeddings = self.model(character_ids)
        tensor = embeddings['elmo_representations'][0]
        return tensor.squeeze()


class FastTextVectorizer(Vectorizer):

    def __init__(self, model_path):
        self.model = load_facebook_model(model_path)

    def embed(self, context):
        return torch.FloatTensor(self.model.wv[context])


class RetrofitVectorizer(Vectorizer):

    def __init__(self, retrofitted_model_path, fasttext_model_path):
        self.model_retrofit = KeyedVectors.load_word2vec_format(
            retrofitted_model_path)
        self.model_fasttext = load_facebook_model(fasttext_model_path)

    def _embed_word(self, word):
        try:
            return torch.FloatTensor(self.model_retrofit[word])
        except KeyError:
            print("Term not found in retrofit model: ", word)
            return torch.FloatTensor(self.model_fasttext.wv[word])

    def embed(self, context):
        tensors = [self._embed_word(word) for word in context]
        return torch.stack(tensors)
