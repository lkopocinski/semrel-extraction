from pathlib import Path
import torch


def save_lines(path: Path, lines, mode='w+'):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode, encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def load_file(path):
    with path.open('r', encoding='utf-8') as f:
        return [line.strip() for line in f]


def save_tensor(path: Path, tensor):
    torch.save(tensor, str(path))
