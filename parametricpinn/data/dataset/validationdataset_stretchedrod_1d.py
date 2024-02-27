from dataclasses import dataclass
from typing import Callable, TypeAlias, cast

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.dataset import (
    ValidationBatch,
    ValidationBatchList,
    ValidationCollateFunc,
)
from parametricpinn.data.geometry import StretchedRod1D
from parametricpinn.types import Tensor

LinearElasticDispalcementSolutionFunc: TypeAlias = Callable[
    [Tensor | float, float, Tensor | float, float, float], Tensor | float
]


def calculate_linear_elastic_displacements_solution(
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
class StretchedRodValidationDatasetLinearElasticity1DConfig:
    length: float
    traction: float
    volume_force: float
    min_youngs_modulus: float
    max_youngs_modulus: float
    num_points: int
    num_samples: int


class StretchedRodValidationDatasetLinearElasticity1D(Dataset):
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
        y_true = calculate_linear_elastic_displacements_solution(
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
