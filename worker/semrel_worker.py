import logging
from pathlib import Path
from typing import Dict, Iterator

import nlp_ws
from corpus_ccl import cclutils

from semrel.data.scripts.corpus import Document
from semrel.data.scripts.utils.io import save_lines
from semrel.data.scripts.vectorizers import ElmoVectorizer, FastTextVectorizer
from semrel.model.scripts import RelNet
from semrel.model.scripts.utils.utils import get_device
from worker.scripts import constant
from worker.scripts.extractor import Parser, find_named_entities, find_nouns
from worker.scripts.prediction import Predictor

_log = logging.getLogger(__name__)


def load_model(model_path: str, vector_size: int = 2648) -> RelNet:
    net = RelNet(in_dim=vector_size)
    net.load(model_path)
    net.eval()
    return net


class SemrelWorker(nlp_ws.NLPWorker):

    @classmethod
    def static_init(cls, config):
        pass

    def init(self):
        _log.critical("Loading models.")
        self._device = get_device()

        _log.critical("Loading ELMO model ...")
        self._elmo = ElmoVectorizer(
            options_path=constant.ELMO_MODEL_OPTIONS,
            weights_path=constant.ELMO_MODEL_WEIGHTS,
            device=self._device.index
        )

        _log.critical("Loading FASTTEXT model ...")
        self._fasttext = FastTextVectorizer(
            model_path=constant.FASTTEXT_MODEL
        )

        _log.critical("Loading models completed.")

    def process(self, input_path: str, task_options: Dict, output_path: str):
        _log.critical("Load MODEL")
        net = load_model(constant.PREDICTION_MODEL)
        net = net.to(self._device)

        if task_options.get(constant.NER_KEY, False):
            parser = Parser(find_named_entities)
        else:
            parser = Parser(find_nouns)

        predictor = Predictor(net, self._elmo, self._fasttext, self._device)

        document = Document(cclutils.read_ccl(input_path))
        for indices_context in parser(document):
            predictions = predictor.predict(indices_context)
            _log.critical(str(predictions))

        # save predictions
        # save_lines(Path(output_path), predictions)


if __name__ == '__main__':
    _log.critical("Start semrel prediction.")
    nlp_ws.NLPService.main(SemrelWorker)
