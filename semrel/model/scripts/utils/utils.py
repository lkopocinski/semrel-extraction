import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List

import torch
import yaml

logger = logging.getLogger(__name__)


def is_better_fscore(fscore: List[int], best_fscore: List[int]) -> bool:
    return fscore[0] > best_fscore[0] and fscore[1] > best_fscore[1]


def is_better_loss(loss: float, best_loss: float) -> bool:
    return loss < best_loss if best_loss else True


def parse_config(path: Path):
    with path.open('r', encoding='utf-8') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exception:
            print(exception)


def get_device() -> torch.device:
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@contextmanager
def ignored(exception):
    """
    Example usage:
        with ignored(Exception):
            do_something()
    """
    try:
        yield
    except exception as e:
        logger.exception("Exception has been raised.", e)
