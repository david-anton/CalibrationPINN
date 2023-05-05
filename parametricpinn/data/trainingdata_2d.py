from collections import namedtuple

import torch

from parametricpinn.data.geometry import Geometry2DProtocol, PlateWithHole
from parametricpinn.data.trainingdata import TrainingDataset
from parametricpinn.types import Tensor

TrainingData2DPDE = namedtuple(
    "TrainingData2DPDE", ["x_coor", "x_E", "x_nu", "f", "y_true"]
)
TrainingData2DStressBC = namedtuple(
    "TrainingData2DStressBC", ["x_coor", "x_E", "x_nu", "normal", "y_true"]
)


class TrainingDataset2D(TrainingDataset):
    def __init__(
        self,
        geometry: Geometry2DProtocol,
        traction: Tensor,
        normal: Tensor,
        volume_force: Tensor,
        min_youngs_modulus: float,
        max_youngs_modulus: float,
        min_poissons_ratio: float,
        max_poissons_ratio: float,
        num_points_pde: int,
        num_points_stress_bc: int,
        num_samples_per_parameter: int,
    ):
        super().__init__()
        self._geometry = geometry
        self._traction = traction
        self._normal = normal
        self._volume_force = volume_force
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._min_poissons_ratio = min_poissons_ratio
        self._max_poissons_ratio = max_poissons_ratio
        self._num_points_pde = num_points_pde
        self._num_points_stress_bc = num_points_stress_bc
        self._num_samples_per_parameter = num_samples_per_parameter
        self._samples_pde: list[TrainingData2DPDE] = []
        self._samples_stress_bc: list[TrainingData2DStressBC] = []

        self._generate_samples()

    def _generate_samples(self) -> None:
        youngs_moduli_list = self._generate_uniform_parameter_list(
            self._min_youngs_modulus, self._max_youngs_modulus
        )
        poissons_ratios_list = self._generate_uniform_parameter_list(
            self._min_poissons_ratio, self._max_poissons_ratio
        )
        for i in range(self._num_samples_per_parameter):
            for j in range(self._num_samples_per_parameter):
                youngs_modulus = youngs_moduli_list[i]
                poissons_ratio = poissons_ratios_list[j]
                self._add_pde_sample(youngs_modulus, poissons_ratio)
                self._add_stress_bc_sample(youngs_modulus, poissons_ratio)

    def _generate_uniform_parameter_list(
        self, min_parameter: float, max_parameter: float
    ) -> list[float]:
        return torch.linspace(
            min_parameter, max_parameter, self._num_samples_per_parameter
        ).tolist()

    def _add_pde_sample(self, youngs_modulus: float, poissons_ratio: float) -> None:
        shape = (self._num_points_pde, 1)
        x_coor = self._geometry.create_random_points(self._num_points_pde)
        x_E = self._repeat_tensor(torch.tensor([youngs_modulus]), shape)
        x_nu = self._repeat_tensor(torch.tensor([poissons_ratio]), shape)
        f = self._repeat_tensor(self._volume_force, shape)
        y_true = torch.zeros((self._num_points_pde, 1), requires_grad=True)
        sample = TrainingData2DPDE(
            x_coor=x_coor, x_E=x_E, x_nu=x_nu, f=f, y_true=y_true
        )
        self._samples_pde.append(sample)

    def _add_stress_bc_sample(
        self, youngs_modulus: float, poissons_ratio: float
    ) -> None:
        shape = (self._num_points_stress_bc, 1)
        x_coor = self._geometry.create_uniform_points_on_left_boundary(
            self._num_points_stress_bc
        )
        x_E = self._repeat_tensor(torch.tensor([youngs_modulus]), shape)
        x_nu = self._repeat_tensor(torch.tensor([poissons_ratio]), shape)
        normal = self._repeat_tensor(self._normal, shape)
        y_true = self._repeat_tensor(self._traction, shape)
        sample = TrainingData2DStressBC(
            x_coor=x_coor, x_E=x_E, x_nu=x_nu, normal=normal, y_true=y_true
        )
        self._samples_stress_bc.append(sample)

    def __len__(self) -> int:
        return self._num_samples_per_parameter**2

    def __getitem__(self, idx: int) -> tuple[TrainingData2DPDE, TrainingData2DStressBC]:
        sample_pde = self._samples_pde[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        return sample_pde, sample_stress_bc


def collate_training_data_2D(
    batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
) -> tuple[TrainingData2DPDE, TrainingData2DStressBC]:
    x_coor_pde_batch = []
    x_E_pde_batch = []
    x_nu_pde_batch = []
    f_pde_batch = []
    y_true_pde_batch = []
    x_coor_stress_bc_batch = []
    x_E_stress_bc_batch = []
    x_nu_stress_bc_batch = []
    normal_stress_bc = []
    y_true_stress_bc_batch = []

    def append_to_pde_batch(sample_pde: TrainingData2DPDE) -> None:
        x_coor_pde_batch.append(sample_pde.x_coor)
        x_E_pde_batch.append(sample_pde.x_E)
        x_nu_pde_batch.append(sample_pde.x_nu)
        f_pde_batch.append(sample_pde.f)
        y_true_pde_batch.append(sample_pde.y_true)

    def append_to_stress_bc_batch(sample_stress_bc: TrainingData2DStressBC) -> None:
        x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
        x_E_stress_bc_batch.append(sample_stress_bc.x_E)
        x_nu_stress_bc_batch.append(sample_stress_bc.x_nu)
        normal_stress_bc.append(sample_stress_bc.normal)
        y_true_stress_bc_batch.append(sample_stress_bc.y_true)

    for sample_pde, sample_stress_bc in batch:
        append_to_pde_batch(sample_pde)
        append_to_stress_bc_batch(sample_stress_bc)

    batch_pde = TrainingData2DPDE(
        x_coor=torch.concat(x_coor_pde_batch, dim=0),
        x_E=torch.concat(x_E_pde_batch, dim=0),
        x_nu=torch.concat(x_nu_pde_batch, dim=0),
        f=torch.concat(f_pde_batch, dim=0),
        y_true=torch.concat(y_true_pde_batch, dim=0),
    )
    batch_stress_bc = TrainingData2DStressBC(
        x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
        x_E=torch.concat(x_E_stress_bc_batch, dim=0),
        x_nu=torch.concat(x_nu_stress_bc_batch, dim=0),
        normal=torch.concat(normal_stress_bc),
        y_true=torch.concat(y_true_stress_bc_batch, dim=0),
    )
    return batch_pde, batch_stress_bc


def create_training_dataset_2D(
    edge_length: float,
    radius: float,
    traction: Tensor,
    normal: Tensor,
    volume_force: Tensor,
    min_youngs_modulus: float,
    max_youngs_modulus: float,
    min_poissons_ratio: float,
    max_poissons_ratio: float,
    num_points_pde: int,
    num_points_stress_bc: int,
    num_samples_per_parameter: int,
):
    geometry = PlateWithHole(edge_length=edge_length, radius=radius)
    return TrainingDataset2D(
        geometry=geometry,
        traction=traction,
        normal=normal,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        min_poissons_ratio=min_poissons_ratio,
        max_poissons_ratio=max_poissons_ratio,
        num_points_pde=num_points_pde,
        num_points_stress_bc=num_points_stress_bc,
        num_samples_per_parameter=num_samples_per_parameter,
    )