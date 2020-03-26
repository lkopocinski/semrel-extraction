import logging
from pathlib import Path
from typing import Dict

import nlp_ws
from corpus_ccl import cclutils

from semrel.data.scripts.corpus import Document
from semrel.data.scripts.utils.io import save_lines
from semrel.data.scripts.vectorizers import ElmoVectorizer
from semrel.model.scripts.utils.utils import get_device
from worker.scripts import constant
from worker.scripts.extractor import \
    Parser, find_named_entities, find_nouns, SentenceWindow
from worker.scripts.prediction import Predictor
from worker.scripts.utils import load_model, format_output

_log = logging.getLogger(__name__)


class SemrelWorker(nlp_ws.NLPWorker):

    @classmethod
    def static_init(cls, config):
        pass

    def init(self):
        self._vector_size = 2048
        self._window_size = 3
        self._device = get_device()
        self._elmo = ElmoVectorizer(
            options_path=constant.ELMO_MODEL_OPTIONS,
            weights_path=constant.ELMO_MODEL_WEIGHTS,
            device=self._device.index
        )

    def process(
            self, input_path: str, task_options: Dict, output_path: str
    ) -> None:
        net = load_model(
            model_path=constant.PREDICTION_MODEL,
            vector_size=self._vector_size,
            device=self._device
        )

        is_ner_task = task_options.get(constant.NER_KEY, False)
        extractor = find_named_entities if is_ner_task else find_nouns

        slicer = SentenceWindow(window_size=self._window_size)
        parser = Parser(slicer, extractor)
        predictor = Predictor(net, self._elmo, self._device)

        document = Document(cclutils.read_ccl(input_path))
        results = [
            predictor.predict(indices_context)
            for indices_context in parser(document)
        ]

        lines = format_output(results)
        save_lines(Path(output_path), lines, mode='a+')


if __name__ == '__main__':
    _log.critical("Start semrel prediction.")
    nlp_ws.NLPService.main(SemrelWorker)
