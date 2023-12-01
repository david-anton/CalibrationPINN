from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.base import generate_uniform_parameter_list
from parametricpinn.data.dataset.dataset import (
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
)
from parametricpinn.data.geometry import PlateWithHole2D
from parametricpinn.types import Tensor

TrainingBatch: TypeAlias = tuple[
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
]
TrainingBatchList: TypeAlias = list[TrainingBatch]
TrainingCollateFunc: TypeAlias = Callable[[TrainingBatchList], TrainingBatch]


@dataclass
class PlateWithHoleTrainingDataset2DConfig:
    plate_length: float
    plate_height: float
    hole_radius: float
    traction_right: Tensor
    volume_force: Tensor
    min_youngs_modulus: float
    max_youngs_modulus: float
    min_poissons_ratio: float
    max_poissons_ratio: float
    num_collocation_points: int
    num_points_per_bc: int
    num_samples_per_parameter: int


class PlateWithHoleTrainingDataset2D(Dataset):
    def __init__(
        self,
        geometry: PlateWithHole2D,
        traction_right: Tensor,
        volume_force: Tensor,
        min_youngs_modulus: float,
        max_youngs_modulus: float,
        min_poissons_ratio: float,
        max_poissons_ratio: float,
        num_collocation_points: int,
        num_points_per_bc: int,
        num_samples_per_parameter: int,
    ):
        super().__init__()
        self._num_parameters = 2
        self._num_traction_bcs = 4
        self._overlap_distance_bcs = 1e-7
        self._overlap_distance_angle_bcs = 1e-7
        self._geometry = geometry
        self._traction_right = traction_right
        self._traction_top = torch.tensor([0.0, 0.0], device=traction_right.device)
        self._traction_bottom = torch.tensor([0.0, 0.0], device=traction_right.device)
        self._traction_hole = torch.tensor([0.0, 0.0], device=traction_right.device)
        self._volume_force = volume_force
        self._min_youngs_modulus = min_youngs_modulus
        self._max_youngs_modulus = max_youngs_modulus
        self._min_poissons_ratio = min_poissons_ratio
        self._max_poissons_ratio = max_poissons_ratio
        self._num_collocation_points = num_collocation_points
        self._num_points_per_bc = num_points_per_bc
        self._num_samples_per_parameter = num_samples_per_parameter
        self._total_num_samples = num_samples_per_parameter**self._num_parameters
        self._samples_collocation: list[TrainingData2DCollocation] = []
        self._samples_traction_bc: list[TrainingData2DTractionBC] = []
        self._generate_samples()

    def get_collate_func(self) -> TrainingCollateFunc:
        def collate_func(batch: TrainingBatchList) -> TrainingBatch:
            x_coor_pde_batch = []
            x_E_pde_batch = []
            x_nu_pde_batch = []
            f_pde_batch = []
            x_coor_traction_bc_batch = []
            x_E_traction_bc_batch = []
            x_nu_traction_bc_batch = []
            normal_traction_bc_batch = []
            area_frac_traction_bc_batch = []
            y_true_traction_bc_batch = []

            def append_to_pde_batch(sample_pde: TrainingData2DCollocation) -> None:
                x_coor_pde_batch.append(sample_pde.x_coor)
                x_E_pde_batch.append(sample_pde.x_E)
                x_nu_pde_batch.append(sample_pde.x_nu)
                f_pde_batch.append(sample_pde.f)

            def append_to_traction_bc_batch(
                sample_traction_bc: TrainingData2DTractionBC,
            ) -> None:
                x_coor_traction_bc_batch.append(sample_traction_bc.x_coor)
                x_E_traction_bc_batch.append(sample_traction_bc.x_E)
                x_nu_traction_bc_batch.append(sample_traction_bc.x_nu)
                normal_traction_bc_batch.append(sample_traction_bc.normal)
                area_frac_traction_bc_batch.append(sample_traction_bc.area_frac)
                y_true_traction_bc_batch.append(sample_traction_bc.y_true)

            for sample_pde, sample_traction_bc in batch:
                append_to_pde_batch(sample_pde)
                append_to_traction_bc_batch(sample_traction_bc)

            batch_pde = TrainingData2DCollocation(
                x_coor=torch.concat(x_coor_pde_batch, dim=0),
                x_E=torch.concat(x_E_pde_batch, dim=0),
                x_nu=torch.concat(x_nu_pde_batch, dim=0),
                f=torch.concat(f_pde_batch, dim=0),
            )
            batch_traction_bc = TrainingData2DTractionBC(
                x_coor=torch.concat(x_coor_traction_bc_batch, dim=0),
                x_E=torch.concat(x_E_traction_bc_batch, dim=0),
                x_nu=torch.concat(x_nu_traction_bc_batch, dim=0),
                normal=torch.concat(normal_traction_bc_batch),
                area_frac=torch.concat(area_frac_traction_bc_batch),
                y_true=torch.concat(y_true_traction_bc_batch, dim=0),
            )
            return batch_pde, batch_traction_bc

        return collate_func

    def _generate_samples(self) -> None:
        youngs_moduli_list = generate_uniform_parameter_list(
            self._min_youngs_modulus,
            self._max_youngs_modulus,
            self._num_samples_per_parameter,
        )
        poissons_ratios_list = generate_uniform_parameter_list(
            self._min_poissons_ratio,
            self._max_poissons_ratio,
            self._num_samples_per_parameter,
        )
        for i in range(self._num_samples_per_parameter):
            for j in range(self._num_samples_per_parameter):
                youngs_modulus = youngs_moduli_list[i]
                poissons_ratio = poissons_ratios_list[j]
                self._add_collocation_sample(youngs_modulus, poissons_ratio)
                self._add_traction_bc_sample(youngs_modulus, poissons_ratio)
                num_sample = i * self._num_samples_per_parameter + j
                print(
                    f"Add training sample {num_sample + 1} / {self._total_num_samples}"
                )

    def _add_collocation_sample(
        self, youngs_modulus: float, poissons_ratio: float
    ) -> None:
        shape = (self._num_collocation_points, 1)
        x_coor = self._geometry.create_random_points(self._num_collocation_points)
        x_E = repeat_tensor(torch.tensor([youngs_modulus]), shape)
        x_nu = repeat_tensor(torch.tensor([poissons_ratio]), shape)
        f = repeat_tensor(self._volume_force, shape)
        sample = TrainingData2DCollocation(
            x_coor=x_coor.detach(), x_E=x_E.detach(), x_nu=x_nu.detach(), f=f.detach()
        )
        self._samples_collocation.append(sample)

    def _add_traction_bc_sample(
        self, youngs_modulus: float, poissons_ratio: float
    ) -> None:
        x_coor, normal = self._create_coordinates_and_normals_for_traction_bcs()
        area_frac = self._calculate_area_fractions_for_traction_bcs()
        x_E, x_nu = self._create_parameters_for_bcs(
            youngs_modulus, poissons_ratio, self._num_traction_bcs
        )
        y_true = self._create_tractions_for_traction_bcs()
        sample = TrainingData2DTractionBC(
            x_coor=x_coor.detach(),
            x_E=x_E.detach(),
            x_nu=x_nu.detach(),
            normal=normal.detach(),
            y_true=y_true.detach(),
            area_frac=area_frac.detach(),
        )
        self._samples_traction_bc.append(sample)

    def _create_coordinates_and_normals_for_traction_bcs(self) -> tuple[Tensor, Tensor]:
        num_points = self._num_points_per_bc
        (
            x_coor_right,
            normal_right,
        ) = self._geometry.create_uniform_points_on_right_boundary(num_points)
        (x_coor_top, normal_top) = self._geometry.create_uniform_points_on_top_boundary(
            num_points
        )
        (
            x_coor_bottom,
            normal_bottom,
        ) = self._geometry.create_uniform_points_on_bottom_boundary(num_points)
        (
            x_coor_hole,
            normal_hole,
        ) = self._geometry.create_uniform_points_on_hole_boundary(num_points)

        self._avoid_overlapping_traction_bcs(
            x_coor_top=x_coor_top, x_coor_bottom=x_coor_bottom
        )
        x_coor = torch.concat(
            (x_coor_right, x_coor_top, x_coor_bottom, x_coor_hole), dim=0
        )
        normal = torch.concat(
            (normal_right, normal_top, normal_bottom, normal_hole), dim=0
        )
        return x_coor, normal

    def _avoid_overlapping_traction_bcs(
        self,
        x_coor_top: Tensor,
        x_coor_bottom: Tensor,
    ) -> None:
        distance = self._overlap_distance_bcs

        def _avoid_overlapping_traction_bcs_at_top_or_bottom(x_coor: Tensor) -> None:
            x_coor[0, 0] += distance
            x_coor[0, -1] -= distance

        _avoid_overlapping_traction_bcs_at_top_or_bottom(x_coor_top)
        _avoid_overlapping_traction_bcs_at_top_or_bottom(x_coor_bottom)

    def _calculate_area_fractions_for_traction_bcs(self) -> Tensor:
        num_points = self._num_points_per_bc
        area_frac_right = self._geometry.calculate_area_fractions_on_vertical_boundary(
            num_points
        )
        area_frac_top = self._geometry.calculate_area_fractions_on_horizontal_boundary(
            num_points
        )
        area_frac_bottom = (
            self._geometry.calculate_area_fractions_on_horizontal_boundary(num_points)
        )
        area_frac_hole = self._geometry.calculate_area_fractions_on_hole_boundary(
            num_points
        )
        return torch.concat(
            (area_frac_right, area_frac_top, area_frac_bottom, area_frac_hole), dim=0
        )

    def _create_parameters_for_bcs(
        self, youngs_modulus: float, poissons_ratio: float, num_bcs: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_bcs * self._num_points_per_bc, 1)
        x_E = repeat_tensor(torch.tensor([youngs_modulus]), shape)
        x_nu = repeat_tensor(torch.tensor([poissons_ratio]), shape)
        return x_E, x_nu

    def _create_tractions_for_traction_bcs(self) -> Tensor:
        shape = (self._num_points_per_bc, 1)
        return torch.concat(
            (
                self._traction_right.repeat(shape),
                self._traction_top.repeat(shape),
                self._traction_bottom.repeat(shape),
                self._traction_hole.repeat(shape),
            ),
            dim=0,
        )

    def __len__(self) -> int:
        return self._num_samples_per_parameter**2

    def __getitem__(
        self, idx: int
    ) -> tuple[TrainingData2DCollocation, TrainingData2DTractionBC]:
        sample_collocation = self._samples_collocation[idx]
        sample_traction_bc = self._samples_traction_bc[idx]
        return sample_collocation, sample_traction_bc
