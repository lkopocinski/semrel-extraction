import logging
import os
from typing import Iterator

import nlp_ws
from corpus_ccl import cclutils

from semrel.data.scripts.corpus import Document
from semrel.data.scripts.vectorizers import ElmoVectorizer
from semrel.model.scripts import RelNet
from semrel.model.scripts.utils.utils import get_device
from worker.extractor import Parser, NounExtractor
from worker.prediction import Predictor

_log = logging.getLogger(__name__)


def load_model(model_path, vector_size=2648) -> RelNet:
    net = RelNet(in_dim=vector_size)
    net.load(model_path)
    net.eval()
    return net


class SemrelWorker(nlp_ws.NLPWorker):

    @classmethod
    def static_init(cls, config):
        pass

    def init(self):
        _log.critical("Started loading models.")
        _log.critical("Loading ELMO ...")

        self.elmo = ElmoVectorizer(
            options=os.getenv('ELMO_MODEL_OPTIONS'),
            weights=os.getenv('ELMO_MODEL_WEIGHTS')
        )

        # _log.critical("Loading FASTTEXT ...")
        self.fasttext = None
        # self.fasttext = FastTextVectorizer(
        #     model_path=os.getenv('FASTTEXT_MODEL')
        # )
        # _log.critical("Finished loading models.")

    def process(self, input_path: str, task_options: dict, output_path: str):
        extractor = self._get_extractor(task_options)
        parser = Parser(extractor)
        predictions = self.predict(input_path, parser)

        self.save_predictions(predictions, output_path)

    def _get_extractor(self, task_options: dict):
        if task_options.get('ner', False):
            extractor = NerExtractor()
        else:
            extractor = NounExtractor()
        return extractor

    def predict(self, path: str, parser: Parser):
        _log.critical("Loading net model ...")
        device = get_device()
        net = load_model(os.getenv('PREDICTION_MODEL'))
        net = net.to(device)
        _log.critical("Net model loaded " + str(net))

        predictor = Predictor(net, self.elmo, self.fasttext, device)

        document = Document(cclutils.read_ccl(path))
        for pair in parser(document):
            decision = predictor.predict(pair)

            (f_idx, f_ctx), (s_idx, s_ctx) = pair
            orth_from = f_ctx[f_idx]
            orth_to = s_ctx[s_idx]

            _log.critical(f'{orth_from}\t{orth_to}: {decision}\n')
            yield f'{orth_from}\t{orth_to}: {decision}\n'

    def save_predictions(self, predictions: Iterator, output_path: str):
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for pred in predictions:
                out_file.write(pred)


if __name__ == '__main__':
    _log.critical("Start semrel prediction.")
    nlp_ws.NLPService.main(SemrelWorker)
