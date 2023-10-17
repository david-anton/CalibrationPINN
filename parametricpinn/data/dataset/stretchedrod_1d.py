from dataclasses import dataclass
from typing import Callable, TypeAlias, cast

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.dataset import (
    TrainingData1DCollocation,
    TrainingData1DTractionBC,
    ValidationBatch,
    ValidationBatchList,
    ValidationCollateFunc,
)
from parametricpinn.data.geometry import StretchedRod1D
from parametricpinn.types import Tensor

TrainingBatch: TypeAlias = tuple[
    TrainingData1DCollocation,
    TrainingData1DTractionBC,
]
TrainingBatchList: TypeAlias = list[TrainingBatch]
TrainingCollateFunc: TypeAlias = Callable[[TrainingBatchList], TrainingBatch]


@dataclass
class StretchedRodTrainingDataset1DConfig:
    length: float
    traction: float
    volume_force: float
    min_youngs_modulus: float
    max_youngs_modulus: float
    num_points_pde: int
    num_samples: int


class StretchedRodTrainingDataset1D(Dataset):
    def __init__(
        self,
        geometry: StretchedRod1D,
        traction: float,
        volume_force: float,
        min_youngs_modulus: float,
        max_youngs_modulus: float,
        num_points_pde: int,
        num_samples: int,
    ):
        super().__init__()
        self._geometry = geometry
        self._traction = traction
        self._volume_force = volume_force
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._num_points_pde = num_points_pde
        self._num_points_stress_bc = 1
        self._num_samples = num_samples
        self._samples_pde: list[TrainingData1DCollocation] = []
        self._samples_stress_bc: list[TrainingData1DTractionBC] = []

        self._generate_samples()

    def get_collate_func(self) -> TrainingCollateFunc:
        def collate_func(batch: TrainingBatchList) -> TrainingBatch:
            x_coor_pde_batch = []
            x_E_pde_batch = []
            f_pde_batch = []
            y_true_pde_batch = []
            x_coor_stress_bc_batch = []
            x_E_stress_bc_batch = []
            y_true_stress_bc_batch = []

            def append_to_pde_batch(sample_pde: TrainingData1DCollocation) -> None:
                x_coor_pde_batch.append(sample_pde.x_coor)
                x_E_pde_batch.append(sample_pde.x_E)
                f_pde_batch.append(sample_pde.f)
                y_true_pde_batch.append(sample_pde.y_true)

            def append_to_stress_bc_batch(
                sample_stress_bc: TrainingData1DTractionBC,
            ) -> None:
                x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
                x_E_stress_bc_batch.append(sample_stress_bc.x_E)
                y_true_stress_bc_batch.append(sample_stress_bc.y_true)

            for sample_pde, sample_stress_bc in batch:
                append_to_pde_batch(sample_pde)
                append_to_stress_bc_batch(sample_stress_bc)

            batch_pde = TrainingData1DCollocation(
                x_coor=torch.concat(x_coor_pde_batch, dim=0),
                x_E=torch.concat(x_E_pde_batch, dim=0),
                f=torch.concat(f_pde_batch, dim=0),
                y_true=torch.concat(y_true_pde_batch, dim=0),
            )
            batch_stress_bc = TrainingData1DTractionBC(
                x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
                x_E=torch.concat(x_E_stress_bc_batch, dim=0),
                y_true=torch.concat(y_true_stress_bc_batch, dim=0),
            )
            return batch_pde, batch_stress_bc

        return collate_func

    def _generate_samples(self) -> None:
        youngs_moduli_list = self._generate_uniform_youngs_modulus_list()
        for i in range(self._num_samples):
            youngs_modulus = youngs_moduli_list[i]
            self._add_pde_sample(youngs_modulus)
            self._add_stress_bc_sample(youngs_modulus)

    def _generate_uniform_youngs_modulus_list(self) -> list[float]:
        return torch.linspace(
            self._min_youngs_modulus, self._max_youngs_modulus, self._num_samples
        ).tolist()

    def _add_pde_sample(self, youngs_modulus: float) -> None:
        shape = (self._num_points_pde, 1)
        x_coor = self._geometry.create_uniform_points(self._num_points_pde)
        x_E = repeat_tensor(torch.tensor([youngs_modulus]), shape)
        f = repeat_tensor(torch.tensor([self._volume_force]), shape)
        y_true = torch.zeros(shape)
        sample = TrainingData1DCollocation(
            x_coor=x_coor.detach(),
            x_E=x_E.detach(),
            f=f.detach(),
            y_true=y_true.detach(),
        )
        self._samples_pde.append(sample)

    def _add_stress_bc_sample(self, youngs_modulus: float) -> None:
        shape = (self._num_points_stress_bc, 1)
        x_coor = self._geometry.create_point_at_free_end()
        x_E = repeat_tensor(torch.tensor([youngs_modulus]), shape)
        y_true = repeat_tensor(torch.tensor([self._traction]), shape)
        sample = TrainingData1DTractionBC(
            x_coor=x_coor.detach(), x_E=x_E.detach(), y_true=y_true.detach()
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


def calculate_displacements_solution(
    coordinates: Tensor | float,
    length: float,
    youngs_modulus: Tensor | float,
    traction: float,
    volume_force: float,
) -> Tensor | float:
    return (traction / youngs_modulus) * coordinates + (
        volume_force / youngs_modulus
    ) * (length * coordinates - 1 / 2 * coordinates**2)


@dataclass
class StretchedRodValidationDataset1DConfig:
    length: float
    traction: float
    volume_force: float
    min_youngs_modulus: float
    max_youngs_modulus: float
    num_points: int
    num_samples: int


class StretchedRodValidationDataset1D(Dataset):
    def __init__(
        self,
        geometry: StretchedRod1D,
        traction: float,
        volume_force: float,
        min_youngs_modulus: float,
        max_youngs_modulus: float,
        num_points: int,
        num_samples: int,
    ) -> None:
        super().__init__()
        self._geometry = geometry
        self._traction = traction
        self._volume_force = volume_force
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._num_points = num_points
        self._num_samples = num_samples
        self._samples_x: list[Tensor] = []
        self._samples_y_true: list[Tensor] = []

        self._generate_samples()

    def get_collate_func(self) -> ValidationCollateFunc:
        def collate_func(batch: ValidationBatchList) -> ValidationBatch:
            x_batch = []
            y_true_batch = []

            for sample_x, sample_y_true in batch:
                x_batch.append(sample_x)
                y_true_batch.append(sample_y_true)

            batch_x = torch.concat(x_batch, dim=0)
            batch_y_true = torch.concat(y_true_batch, dim=0)
            return batch_x, batch_y_true

        return collate_func

    def _generate_samples(self) -> None:
        for i in range(self._num_samples):
            youngs_modulus = self._generate_random_youngs_modulus()
            coordinates = self._generate_random_coordinates()
            self._add_input_sample(coordinates, youngs_modulus)
            self._add_output_sample(coordinates, youngs_modulus)

    def _generate_random_youngs_modulus(self) -> Tensor:
        return self._min_youngs_modulus + torch.rand((1)) * (
            self._max_youngs_modulus - self._min_youngs_modulus
        )

    def _generate_random_coordinates(self) -> Tensor:
        return self._geometry.create_random_points(self._num_points)

    def _add_input_sample(self, coordinates: Tensor, youngs_modulus: Tensor) -> None:
        x_coor = coordinates
        x_E = repeat_tensor(youngs_modulus, (self._num_points, 1))
        x = torch.concat((x_coor, x_E), dim=1)
        self._samples_x.append(x)

    def _add_output_sample(self, coordinates: Tensor, youngs_modulus: Tensor) -> None:
        y_true = calculate_displacements_solution(
            coordinates=coordinates,
            length=self._geometry.length,
            youngs_modulus=youngs_modulus,
            traction=self._traction,
            volume_force=self._volume_force,
        )
        self._samples_y_true.append(cast(Tensor, y_true))

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample_x = self._samples_x[idx]
        sample_y_true = self._samples_y_true[idx]
        return sample_x, sample_y_true
