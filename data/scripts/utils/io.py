from pathlib import Path
from typing import Generator

import torch


def save_lines(path: Path, lines, mode='w+'):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode, encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def read_lines(path: Path) -> Generator[str]:
    with path.open('r', encoding='utf-8') as f:
        return (line.strip() for line in f)


def save_tensor(path: Path, tensor):
    torch.save(tensor, str(path))
