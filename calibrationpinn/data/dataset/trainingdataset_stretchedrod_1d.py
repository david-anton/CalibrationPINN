from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
from torch.utils.data import Dataset

from calibrationpinn.data.base import repeat_tensor
from calibrationpinn.data.dataset.dataset import (
    TrainingData1DCollocation,
    TrainingData1DTractionBC,
)
from calibrationpinn.data.geometry import StretchedRod1D
from calibrationpinn.types import Tensor

TrainingBatch: TypeAlias = tuple[
    TrainingData1DCollocation,
    TrainingData1DTractionBC,
]
TrainingBatchList: TypeAlias = list[TrainingBatch]
TrainingCollateFunc: TypeAlias = Callable[[TrainingBatchList], TrainingBatch]


@dataclass
class StretchedRodTrainingDataset1DConfig:
    parameters_samples: Tensor
    length: float
    traction: float
    volume_force: float
    num_points_pde: int


class StretchedRodTrainingDataset1D(Dataset):
    def __init__(
        self,
        parameters_samples: Tensor,
        geometry: StretchedRod1D,
        traction: float,
        volume_force: float,
        num_points_pde: int,
    ):
        super().__init__()
        self._parameters_samples = parameters_samples
        self._geometry = geometry
        self._traction = traction
        self._volume_force = volume_force
        self._num_points_pde = num_points_pde
        self._num_points_stress_bc = 1
        self._num_samples = len(self._parameters_samples)
        self._samples_pde: list[TrainingData1DCollocation] = []
        self._samples_stress_bc: list[TrainingData1DTractionBC] = []
        self._generate_samples()

    def get_collate_func(self) -> TrainingCollateFunc:
        def collate_func(batch: TrainingBatchList) -> TrainingBatch:
            x_coor_pde_batch = []
            x_params_pde_batch = []
            f_pde_batch = []
            x_coor_stress_bc_batch = []
            x_params_stress_bc_batch = []
            y_true_stress_bc_batch = []

            def append_to_pde_batch(sample_pde: TrainingData1DCollocation) -> None:
                x_coor_pde_batch.append(sample_pde.x_coor)
                x_params_pde_batch.append(sample_pde.x_params)
                f_pde_batch.append(sample_pde.f)

            def append_to_stress_bc_batch(
                sample_stress_bc: TrainingData1DTractionBC,
            ) -> None:
                x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
                x_params_stress_bc_batch.append(sample_stress_bc.x_params)
                y_true_stress_bc_batch.append(sample_stress_bc.y_true)

            for sample_pde, sample_stress_bc in batch:
                append_to_pde_batch(sample_pde)
                append_to_stress_bc_batch(sample_stress_bc)

            batch_pde = TrainingData1DCollocation(
                x_coor=torch.concat(x_coor_pde_batch, dim=0),
                x_params=torch.concat(x_params_pde_batch, dim=0),
                f=torch.concat(f_pde_batch, dim=0),
            )
            batch_stress_bc = TrainingData1DTractionBC(
                x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
                x_params=torch.concat(x_params_stress_bc_batch, dim=0),
                y_true=torch.concat(y_true_stress_bc_batch, dim=0),
            )
            return batch_pde, batch_stress_bc

        return collate_func

    def _generate_samples(self) -> None:
        for _, parameters_sample in enumerate(self._parameters_samples):
            self._add_pde_sample(parameters_sample)
            self._add_stress_bc_sample(parameters_sample)

    def _add_pde_sample(self, parameters_sample: Tensor) -> None:
        shape = (self._num_points_pde, 1)
        x_coor = self._geometry.create_uniform_points(self._num_points_pde)
        x_params = repeat_tensor(parameters_sample, shape)
        f = repeat_tensor(torch.tensor([self._volume_force]), shape)
        sample = TrainingData1DCollocation(
            x_coor=x_coor.detach(),
            x_params=x_params.detach(),
            f=f.detach(),
        )
        self._samples_pde.append(sample)

    def _add_stress_bc_sample(self, parameters_sample: Tensor) -> None:
        shape = (self._num_points_stress_bc, 1)
        x_coor = self._geometry.create_point_at_free_end()
        x_params = repeat_tensor(parameters_sample, shape)
        y_true = repeat_tensor(torch.tensor([self._traction]), shape)
        sample = TrainingData1DTractionBC(
            x_coor=x_coor.detach(), x_params=x_params.detach(), y_true=y_true.detach()
        )
        self._samples_stress_bc.append(sample)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[TrainingData1DCollocation, TrainingData1DTractionBC]:
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc
