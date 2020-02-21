import os

import nlp_ws
import torch
from corpus_ccl import cclutils

from data.scripts.utils.vectorizers import ElmoVectorizer, FastTextVectorizer
from worker.extractor import Parser, NounExtractor, NERxtractor
from scripts import RelNet


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model_path, vector_size=2648) -> RelNet:
    net = RelNet(in_dim=vector_size)
    net.load(model_path)
    net.eval()
    return net


def documents(fileindex: str):
    with open(fileindex, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if os.path.exists(line.strip())]
    return (cclutils.read_ccl(path) for path in paths)


class Predictor(object):

    def __init__(self, net_model, elmo, fasttext, device='cpu'):
        self._net = net_model
        self._elmo = elmo
        self._fasttext = fasttext
        self.device = device

    def _make_vectors(self, pair):
        (idx1, ctx1), (idx2, ctx2) = pair
        print(pair)
        ev1 = self._elmo.embed(ctx1)[idx1]
        ev2 = self._elmo.embed(ctx2)[idx2]

        fv1 = self._fasttext.embed(ctx1)[idx1]
        fv2 = self._fasttext.embed(ctx2)[idx2]

        v = torch.cat([ev1, ev2, fv1, fv2])
        return v.to(self.device)

    def _predict(self, vectors):
        with torch.no_grad():
            prediction = self._net(vectors)
            prediction = torch.argmax(prediction)
            return prediction.item()

    def predict(self, pair):
        vectors = self._make_vectors(pair)
        return self._predict(vectors)


class SemrelWorker(nlp_ws.NLPWorker):

    @classmethod
    def static_init(cls, config):
        cls.elmo = ElmoVectorizer(
            options_path=os.getenv('ELMO_MODEL_OPTIONS'),
            weights_path=os.getenv('ELMO_MODEL_WEIGHTS')
        )

        cls.fasttext = FastTextVectorizer(
            model_path=os.getenv('FASTTEXT_MODEL')
        )

    def process(self, input_path, task_options, output_path):
        if task_options.get('ner', False):
            extractor = NERxtractor()
        else:
            extractor = NounExtractor()

        predictions = self.predict(input_path, Parser(extractor))

        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(pred)

    def predict(self, fileindex, parser):
        device = get_device()
        net = load_model(os.getenv('PREDICTION_MODEL'))
        net = net.to(device)

        predictor = Predictor(net, self.elmo, self.fasttext, device)

        for doc in documents(fileindex):
            for pair in parser(doc):
                decision = predictor.predict(pair)
                (f_idx, f_ctx), (s_idx, s_ctx) = pair

                orth_from = f_ctx[f_idx]
                orth_to = s_ctx[s_idx]
                yield f'{orth_from}\t{orth_to}: {decision}\n'


if __name__ == '__main__':
    print('Start')
    nlp_ws.NLPService.main(SemrelWorker)
