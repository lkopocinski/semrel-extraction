import os

import nlp_ws
import torch
from corpus_ccl import cclutils

from data.scripts.utils.vectorizers import ElmoVectorizer, FastTextVectorizer
from relextr.evaluation.scripts.extractor import Parser
from relextr.model.scripts.relnet import RelNet


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

    def __init__(self, net_model, elmo, fasttext):
        self._net = net_model
        self._elmo = elmo
        self._fasttext = fasttext
        self.device = 'cpu'

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
        predictions = self._predict(vectors)
        return predictions


class SemrelWorker(nlp_ws.NLPWorker):

    def process(self, input_path, task_options, output_path):
        if task_options.get('ner', False):
            # TODO:
            pass
        else:
            out = self.predict(input_path)

    def predict(self, fileindex):
        device = get_device()
        net = load_model(
            model_path=os.getenv('PREDICTION_MODEL')
        )
        net = net.to(device)

        elmo = ElmoVectorizer(
            options=os.getenv('ELMO_MODEL_OPTIONS'),
            weights=os.getenv('ELMO_MODEL_WEIGHTS')
        )

        fasttext = FastTextVectorizer(
            model_path=os.getenv('FASTTEXT_MODEL')
        )

        predictor = Predictor(net, elmo, fasttext)
        predictor.device = device
        parser = Parser()

        for doc in documents(fileindex):
            for pair in parser(doc):
                decision = predictor.predict(pair)
                (f_idx, f_ctx), (s_idx, s_ctx) = pair
                yield f'{f_ctx[f_idx]}\t{s_ctx[s_idx]}: {decision}\n'


if __name__ == '__main__':
    print('Start')
    nlp_ws.NLPService.main(SemrelWorker)
