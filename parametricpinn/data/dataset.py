from typing import NamedTuple, TypeAlias

import torch

from parametricpinn.types import Tensor

TrainingData: TypeAlias = NamedTuple


class Dataset(torch.utils.data.Dataset):
    def _repeat_tensor(self, tensor: Tensor, dim: tuple[int, ...]) -> Tensor:
        return tensor.repeat(dim)
