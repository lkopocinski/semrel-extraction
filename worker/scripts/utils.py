from typing import List

import torch

from semrel.model.scripts import RelNet
from worker.scripts.prediction import Results


def load_model(
        model_path: str, vector_size: int, device: torch.device
) -> RelNet:
    net = RelNet(in_dim=vector_size)
    net.load(model_path)
    net = net.to(device)
    net.eval()
    return net


def format_output(
        results: List[Results]
) -> List[str]:
    return [
        f'{orth_from} : {orth_to} - {prediction}'
        for orths, predictions in results
        for (orth_from, orth_to), prediction in zip(orths, predictions)
    ]
