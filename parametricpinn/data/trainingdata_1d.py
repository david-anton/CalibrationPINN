from collections import namedtuple

import torch

from parametricpinn.data.geometry import Geometry1DProtocol, StretchedRod
from parametricpinn.data.dataset import Dataset
from parametricpinn.types import Tensor

TrainingData1DPDE = namedtuple("TrainingData1DPDE", ["x_coor", "x_E", "f", "y_true"])
TrainingData1DStressBC = namedtuple(
    "TrainingData1DStressBC", ["x_coor", "x_E", "y_true"]
)


class TrainingDataset1D(Dataset):
    def __init__(
        self,
        geometry: Geometry1DProtocol,
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
        self._samples_pde: list[TrainingData1DPDE] = []
        self._samples_stress_bc: list[TrainingData1DStressBC] = []

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
        shape = (self._num_points_pde, 1)
        x_coor = self._geometry.create_uniform_points(self._num_points_pde)
        x_E = self._repeat_tensor(torch.tensor([youngs_modulus]), shape)
        f = self._repeat_tensor(torch.tensor([self._volume_force]), shape)
        y_true = torch.zeros(shape, requires_grad=True)
        sample = TrainingData1DPDE(x_coor=x_coor, x_E=x_E, f=f, y_true=y_true)
        self._samples_pde.append(sample)

    def _add_stress_bc_sample(self, youngs_modulus: float) -> None:
        shape = (self._num_points_stress_bc, 1)
        x_coor = self._geometry.create_point_at_free_end()
        x_E = self._repeat_tensor(torch.tensor([youngs_modulus]), shape)
        y_true = self._repeat_tensor(torch.tensor([self._traction]), shape)
        sample = TrainingData1DStressBC(x_coor=x_coor, x_E=x_E, y_true=y_true)
        self._samples_stress_bc.append(sample)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[TrainingData1DPDE, TrainingData1DStressBC]:
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc


def collate_training_data_1D(
    batch: list[tuple[TrainingData1DPDE, TrainingData1DStressBC]]
) -> tuple[TrainingData1DPDE, TrainingData1DStressBC]:
    x_coor_pde_batch = []
    x_E_pde_batch = []
    f_pde_batch = []
    y_true_pde_batch = []
    x_coor_stress_bc_batch = []
    x_E_stress_bc_batch = []
    y_true_stress_bc_batch = []

    def append_to_pde_batch(sample_pde: TrainingData1DPDE) -> None:
        x_coor_pde_batch.append(sample_pde.x_coor)
        x_E_pde_batch.append(sample_pde.x_E)
        f_pde_batch.append(sample_pde.f)
        y_true_pde_batch.append(sample_pde.y_true)

    def append_to_stress_bc_batch(sample_stress_bc: TrainingData1DStressBC) -> None:
        x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
        x_E_stress_bc_batch.append(sample_stress_bc.x_E)
        y_true_stress_bc_batch.append(sample_stress_bc.y_true)

    for sample_pde, sample_stress_bc in batch:
        append_to_pde_batch(sample_pde)
        append_to_stress_bc_batch(sample_stress_bc)

    batch_pde = TrainingData1DPDE(
        x_coor=torch.concat(x_coor_pde_batch, dim=0),
        x_E=torch.concat(x_E_pde_batch, dim=0),
        f=torch.concat(f_pde_batch, dim=0),
        y_true=torch.concat(y_true_pde_batch, dim=0),
    )
    batch_stress_bc = TrainingData1DStressBC(
        x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
        x_E=torch.concat(x_E_stress_bc_batch, dim=0),
        y_true=torch.concat(y_true_stress_bc_batch, dim=0),
    )
    return batch_pde, batch_stress_bc


def create_training_dataset_1D(
    length: float,
    traction: float,
    volume_force: float,
    min_youngs_modulus: float,
    max_youngs_modulus: float,
    num_points_pde: int,
    num_samples: int,
):
    geometry = StretchedRod(length=length)
    return TrainingDataset1D(
        geometry=geometry,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples,
    )
