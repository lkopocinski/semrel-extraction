from pathlib import Path

import yaml
import torch


def is_better_fscore(fscore, best_fscore):
    return fscore[0] > best_fscore[0] and fscore[1] > best_fscore[1]


def parse_config(path: Path):
    with path.open('r', encoding='utf-8') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def get_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Runing on: {device}.')
    return device