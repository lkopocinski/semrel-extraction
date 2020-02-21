import json
from pathlib import Path
from typing import Iterator

import torch


def save_line(path: Path, line: str, mode='w+'):
    save_lines(path, [line], mode)


def save_lines(path: Path, lines: Iterator, mode='w+'):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode, encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def read_lines(path: Path) -> Iterator[str]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def save_tensor(path: Path, tensor: torch.Tensor):
    torch.save(tensor, str(path))


def save_json(to_save_content, save_path: Path):
    with save_path.open("w", encoding='utf-8') as file:
        json.dump(to_save_content, file)
