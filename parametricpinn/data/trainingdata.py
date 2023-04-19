# Standard library imports
from typing import NamedTuple, TypeAlias

# Third-party imports
import torch
from torch.utils.data import Dataset

# Local library imports
from parametricpinn.types import Tensor

TrainingData: TypeAlias = NamedTuple


class TrainingDataset(Dataset):
    def _generate_full_tensor(self, fill_value: float, num_points: int) -> Tensor:
        return torch.full((num_points, 1), fill_value, requires_grad=True)
