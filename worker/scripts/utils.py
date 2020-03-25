from typing import List, Tuple

import numpy
import torch

from semrel.model.scripts import RelNet


def load_model(
        model_path: str, vector_size: int, device: torch.device
) -> RelNet:
    net = RelNet(in_dim=vector_size)
    net.load(model_path)
    net = net.to(device)
    net.eval()
    return net


def format_output(
        orths: List[Tuple[str, str]], predictions: numpy.array
) -> List[str]:
    return [
        f'{orth_from} : {orth_to} - {prediction}'
        for (orth_from, orth_to), prediction in zip(orths, predictions)
    ]
