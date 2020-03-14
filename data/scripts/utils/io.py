import json
from pathlib import Path
from typing import Iterator, Dict, Generator

import torch


def save_line(path: Path, line: str, mode='w+'):
    save_lines(path, [line], mode)


def save_lines(path: Path, lines: Iterator, mode='w+'):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode, encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def read_lines(path: Path) -> Generator[str]:
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()


def save_tensor(path: Path, tensor: torch.Tensor):
    torch.save(tensor, str(path))


def save_json(to_save_content, path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding='utf-8') as file:
        json.dump(to_save_content, file)


def load_json(indices_file: Path) -> Dict:
    with indices_file.open('r', encoding='utf-8') as file:
        return json.load(file)
