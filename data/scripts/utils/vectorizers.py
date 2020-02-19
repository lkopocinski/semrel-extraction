import abc
from typing import List

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model


class Vectorizer(abc.ABC):

    @abc.abstractmethod
    def embed(self, context: List[str]) -> torch.Tensor:
        pass


class ElmoVectorizer(Vectorizer):

    def __init__(self, options_path: str, weights_path: str):
        self.model = Elmo(options_path, weights_path, 1, dropout=0)

    def embed(self, context: List[str]) -> torch.Tensor:
        character_ids = batch_to_ids([context])
        embeddings = self.model(character_ids)
        tensor = embeddings['elmo_representations'][0]
        return tensor.squeeze()


class FastTextVectorizer(Vectorizer):

    def __init__(self, model_path: str):
        self.model = load_facebook_model(model_path)

    def embed(self, context: List[str]) -> torch.Tensor:
        return torch.from_numpy(self.model.wv[context])


class RetrofitVectorizer(Vectorizer):

    def __init__(self, retrofit_model_path, fasttext_model_path):
        self.model_retrofit = KeyedVectors.load_word2vec_format(retrofit_model_path)
        self.model_fasttext = load_facebook_model(fasttext_model_path)

    def _embed_word(self, word: str) -> torch.Tensor:
        try:
            return torch.from_numpy(self.model_retrofit[word])
        except KeyError:
            print("Term not found in retrofit model: ", word)
            return torch.from_numpy(self.model_fasttext.wv[word])

    def embed(self, context: List[str]) -> torch.Tensor:
        tensors = [self._embed_word(word) for word in context]
        return torch.stack(tensors)
