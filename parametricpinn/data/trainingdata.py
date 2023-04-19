# Standard library imports
from typing import NamedTuple, TypeAlias

# Third-party imports
import torch
from torch.utils.data import Dataset

# Local library imports
from parametricpinn.types import Tensor

TrainingData: TypeAlias = NamedTuple


class TrainingDataset(Dataset):
    def __init__(self) -> None:
        self._samples_pde: list[TrainingData] = []
        self._samples_stress_bc: list[TrainingData] = []

    def _generate_full_tensor(self, fill_value: float, num_points: int) -> Tensor:
        return torch.full((num_points, 1), fill_value, requires_grad=True)

    def __getitem__(self, idx: int) -> tuple[TrainingData, TrainingData]:
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc
