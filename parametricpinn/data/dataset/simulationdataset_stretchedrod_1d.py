from dataclasses import dataclass
from typing import Callable, TypeAlias, cast

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.dataset import (
    SimulationCollateFunc,
    SimulationData,
    SimulationDataList,
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
class StretchedRodSimulationDatasetLinearElasticity1DConfig:
    length: float
    traction: float
    volume_force: float
    min_youngs_modulus: float
    max_youngs_modulus: float
    num_points: int
    num_samples: int


class StretchedRodSimulationDatasetLinearElasticity1D(Dataset):
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
        self._samples: SimulationDataList = []

        self._generate_samples()

    def get_collate_func(self) -> SimulationCollateFunc:
        def collate_func(batch: SimulationDataList) -> SimulationData:
            x_coor_batch = []
            x_params_batch = []
            y_true_batch = []

            def append_to_batch(sample: SimulationData) -> None:
                x_coor_batch.append(sample.x_coor)
                x_params_batch.append(sample.x_params)
                y_true_batch.append(sample.y_true)

            for sample in batch:
                append_to_batch(sample)

            batched_data = SimulationData(
                x_coor=torch.concat(x_coor_batch, dim=0),
                x_params=torch.concat(x_params_batch, dim=0),
                y_true=torch.concat(y_true_batch, dim=0),
            )
            return batched_data

        return collate_func

    def _generate_samples(self) -> None:
        for i in range(self._num_samples):
            youngs_modulus = self._generate_random_youngs_modulus()
            coordinates = self._generate_random_coordinates()
            self._add_sample(coordinates, youngs_modulus)

    def _generate_random_youngs_modulus(self) -> Tensor:
        return self._min_youngs_modulus + torch.rand((1)) * (
            self._max_youngs_modulus - self._min_youngs_modulus
        )

    def _generate_random_coordinates(self) -> Tensor:
        return self._geometry.create_random_points(self._num_points)

    def _add_sample(self, coordinates: Tensor, youngs_modulus: Tensor) -> None:
        x_coor = coordinates
        x_params = repeat_tensor(youngs_modulus, (self._num_points, 1))
        y_true = calculate_linear_elastic_displacements_solution(
            coordinates=coordinates,
            length=self._geometry.length,
            youngs_modulus=youngs_modulus,
            traction=self._traction,
            volume_force=self._volume_force,
        )
        sample = SimulationData(x_coor=x_coor, x_params=x_params, y_true=y_true)
        self._samples.append(sample)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> SimulationData:
        return self._samples[idx]
