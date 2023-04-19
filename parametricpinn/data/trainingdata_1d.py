# Standard library imports
from collections import namedtuple

# Third-party imports
import torch
from torch.utils.data import Dataset

# Local library imports
from parametricpinn.data.geometry import StretchedRod
from parametricpinn.data.trainingdata import TrainingDataset
from parametricpinn.types import Tensor


TrainingData1D = namedtuple("TrainingData1D", ["x_coor", "x_E", "y_true"])


class TrainingDataset1D(TrainingDataset):
    def __init__(
        self,
        geometry: StretchedRod,
        traction: float,
        min_youngs_modulus: float,
        max_youngs_modulus: float,
        num_points_pde: int,
        num_samples: int,
    ):
        super().__init__()
        self._geometry = geometry
        self._traction = traction
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._num_points_pde = num_points_pde
        self._num_points_stress_bc = 1
        self._num_samples = num_samples

        self._generate_samples()

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
        x_coor = self._geometry.create_uniform_points(self._num_points_pde)
        x_E = self._generate_full_tensor(youngs_modulus, self._num_points_pde)
        y_true = torch.zeros_like(x_coor, requires_grad=True)
        sample = TrainingData1D(x_coor=x_coor, x_E=x_E, y_true=y_true)
        self._samples_pde.append(sample)

    def _add_stress_bc_sample(self, youngs_modulus: float) -> None:
        x_coor = self._geometry.create_points_at_free_end(self._num_points_stress_bc)
        x_E = self._generate_full_tensor(youngs_modulus, self._num_points_stress_bc)
        y_true = self._generate_full_tensor(self._traction, self._num_points_stress_bc)
        sample = TrainingData1D(x_coor=x_coor, x_E=x_E, y_true=y_true)
        self._samples_stress_bc.append(sample)

    def __len__(self) -> int:
        return self._num_samples


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


def create_training_dataset_1D(
    length: float,
    traction: float,
    min_youngs_modulus: float,
    max_youngs_modulus: float,
    num_points_pde: int,
    num_samples: int,
):
    geometry = StretchedRod(length=length)
    return TrainingDataset1D(
        geometry=geometry,
        traction=traction,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples,
    )
