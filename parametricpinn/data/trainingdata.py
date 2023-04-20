from typing import NamedTuple, TypeAlias

import torch
from torch.utils.data import Dataset

from parametricpinn.types import Tensor

TrainingData: TypeAlias = NamedTuple


class TrainingDataset(Dataset):
    def _repeat_tensor(self, tensor: Tensor, dim: tuple[int, ...]) -> Tensor:
        return tensor.repeat(dim)
