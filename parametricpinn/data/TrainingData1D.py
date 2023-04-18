# Standard library imports
from collections import namedtuple

# Third-party imports
import torch
from torch.utils.data import Dataset

# Local library imports


TrainingData1D = namedtuple("TrainingData1D", ["x_coor", "x_E", "y_true"])


class TrainingDataset1D(Dataset):
    def __init__(
        self,
        length: float,
        traction: float,
        min_youngs_modulus: float,
        max_youngs_modulus: float,
        num_points_pde: int,
        num_samples: int,
    ):
        self._length = length
        self._traction = traction
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._num_points_pde = num_points_pde
        self._num_points_stress_bc = 1
        self._num_samples = num_samples
        self._samples_pde: list[TrainingData1D] = []
        self._samples_stress_bc: list[TrainingData1D] = []

        self._generate_samples()

    def _generate_samples(self) -> None:
        youngs_modulus_list = self._generate_uniform_youngs_modulus_list()
        for i in range(self._num_samples):
            youngs_modulus = youngs_modulus_list[i]
            self._add_pde_sample(youngs_modulus)
            self._add_stress_bc_sample(youngs_modulus)

    def _generate_uniform_youngs_modulus_list(self) -> list[float]:
        return torch.linspace(
            self._min_youngs_modulus, self._max_youngs_modulus, self._num_samples
        ).tolist()

    def _add_pde_sample(self, youngs_modulus: float) -> None:
        x_coor = torch.linspace(
            0.0, self._length, self._num_points_pde, requires_grad=True
        ).view(self._num_points_pde, 1)
        x_E = torch.full((self._num_points_pde, 1), youngs_modulus)
        y_true = torch.zeros_like(x_coor)
        sample = TrainingData1D(x_coor=x_coor, x_E=x_E, y_true=y_true)
        self._samples_pde.append(sample)

    def _add_stress_bc_sample(self, youngs_modulus: float) -> None:
        x_coor = torch.full(
            (self._num_points_stress_bc, 1), self._length, requires_grad=True
        )
        x_E = torch.full((self._num_points_stress_bc, 1), youngs_modulus)
        y_true = torch.full((self._num_points_stress_bc, 1), self._traction)
        sample = TrainingData1D(x_coor=x_coor, x_E=x_E, y_true=y_true)
        self._samples_stress_bc.append(sample)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[TrainingData1D, TrainingData1D]:
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc


def collate_training_data_1D(
    batch: list[tuple[TrainingData1D, TrainingData1D]]
) -> tuple[TrainingData1D, TrainingData1D]:
    x_coor_pde_batch = []
    x_E_pde_batch = []
    y_true_pde_batch = []
    x_coor_stress_bc_batch = []
    x_E_stress_bc_batch = []
    y_true_stress_bc_batch = []

    def append_to_pde_batch(sample_pde: TrainingData1D) -> None:
        x_coor_pde_batch.append(sample_pde.x_coor)
        x_E_pde_batch.append(sample_pde.x_E)
        y_true_pde_batch.append(sample_pde.y_true)

    def append_to_stress_bc_batch(sample_stress_bc: TrainingData1D) -> None:
        x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
        x_E_stress_bc_batch.append(sample_stress_bc.x_E)
        y_true_stress_bc_batch.append(sample_stress_bc.y_true)

    for sample_pde, sample_stress_bc in batch:
        append_to_pde_batch(sample_pde)
        append_to_stress_bc_batch(sample_stress_bc)

    batch_pde = TrainingData1D(
        x_coor=torch.concat(x_coor_pde_batch, dim=0),
        x_E=torch.concat(x_E_pde_batch, dim=0),
        y_true=torch.concat(y_true_pde_batch, dim=0),
    )
    batch_stress_bc = TrainingData1D(
        x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
        x_E=torch.concat(x_E_stress_bc_batch, dim=0),
        y_true=torch.concat(y_true_stress_bc_batch, dim=0),
    )
    return batch_pde, batch_stress_bc
