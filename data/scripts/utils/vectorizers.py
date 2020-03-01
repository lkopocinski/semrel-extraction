import abc
from typing import List

import sent2vec
import torch
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.modules.elmo import Elmo, batch_to_ids
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model
import numpy as np


class Vectorizer(abc.ABC):

    @abc.abstractmethod
    def embed(self, context: List[str]) -> torch.Tensor:
        pass


class ElmoEmbedderVectorizer(Vectorizer):

    def __init__(self, options_path: str, weights_path: str):
        self.model = ElmoEmbedder(options_path, weights_path, cuda_device=0)

    def embed(self, context: List[str]) -> torch.Tensor:
#        import pudb; pudb.set_trace()
        vectors = self.model.embed_sentence(context)
        vectors = np.average(vectors, axis=0)
        tensor = torch.from_numpy(vectors)
        return tensor


class ElmoVectorizer(Vectorizer):

    def __init__(self, options_path: str, weights_path: str):
        self.model = Elmo(options_path, weights_path, 1, dropout=0)
        self.model = self.model.to("cuda:0")

    def embed(self, context: List[str]) -> torch.Tensor:
        character_ids = batch_to_ids([context])
        character_ids = character_ids.to('cuda:0')
        embeddings = self.model(character_ids)

        character_ids.to('cpu')
        embeddings = embeddings.to('cpu')

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


class Sent2VecVectorizer(Vectorizer):

    def __init__(self, model_path: str):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(model_path, inference_mode=True)

    def embed(self, sentence: List[str]) -> torch.Tensor:
        sentence = ' '.join(sentence)
        return torch.from_numpy(self.model.embed_sentence(sentence))
