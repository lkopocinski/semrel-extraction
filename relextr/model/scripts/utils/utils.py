import torch
import yaml

from contextlib import contextmanager


def is_better_fscore(fscore, best_fscore):
    return fscore[0] > best_fscore[0] and fscore[1] > best_fscore[1]


def is_better_loss(loss, best_loss):
    return loss < best_loss if best_loss else True


def parse_config(path):
    with open(path, 'r', encoding='utf-8') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Runing on: {device}.')
    return device


@contextmanager
def ignored(exception):
    """
    A simple try/except wrapper useful for scraping.
    Example usage:
        with ignored(Exception):
            do_something()
    """
    try:
        yield
    except exception as e:
        logger.exception("Exception has been raised.", e)
