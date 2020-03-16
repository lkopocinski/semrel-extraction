from typing import List

import torch
from torch import nn

from data.scripts.constant import PHRASE_LENGTH_LIMIT


def max_pool_member_vectors(tensor: torch.Tensor) -> torch.Tensor:
    pool = nn.MaxPool1d(PHRASE_LENGTH_LIMIT, stride=0)
    tensor = tensor.transpose(2, 1)
    output = pool(tensor)
    return output.transpose(2, 1).squeeze()


def max_pool_relation_vectors(vectors: List) -> torch.Tensor:
    members_from_vectors, member_to_vectors = zip(*vectors)

    members_from_vectors = torch.cat(members_from_vectors)
    member_to_vectors = torch.cat(member_to_vectors)

    pooled1 = max_pool_member_vectors(members_from_vectors)
    pooled2 = max_pool_member_vectors(member_to_vectors)

    return torch.cat([pooled1, pooled2], dim=1)
