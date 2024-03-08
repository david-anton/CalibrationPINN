import math
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.dataset import (
    TrainingData2DCollocation,
    TrainingData2DStressBC,
    TrainingData2DTractionBC,
)
from parametricpinn.data.geometry import QuarterPlateWithHole2D
from parametricpinn.types import Tensor

TrainingBatch: TypeAlias = tuple[
    TrainingData2DCollocation,
    TrainingData2DStressBC,
    TrainingData2DTractionBC,
]
TrainingBatchList: TypeAlias = list[TrainingBatch]
TrainingCollateFunc: TypeAlias = Callable[[TrainingBatchList], TrainingBatch]


@dataclass
class QuarterPlateWithHoleTrainingDataset2DConfig:
    parameters_samples: Tensor
    edge_length: float
    radius: float
    traction_left: Tensor
    volume_force: Tensor
    num_collocation_points: int
    num_points_per_bc: int
    bcs_overlap_distance: float
    bcs_overlap_angle_distance: float


class QuarterPlateWithHoleTrainingDataset2D(Dataset):
    def __init__(
        self,
        parameters_samples: Tensor,
        geometry: QuarterPlateWithHole2D,
        traction_left: Tensor,
        volume_force: Tensor,
        num_collocation_points: int,
        num_points_per_bc: int,
        bcs_overlap_distance: float,
        bcs_overlap_angle_distance: float,
    ):
        super().__init__()
        self._num_stress_bcs = 2
        self._num_traction_bcs = 3
        self._bcs_overlap_distance = bcs_overlap_distance
        self._bcs_overlap_angle_distance = bcs_overlap_angle_distance
        self._parameters_samples = parameters_samples
        self._geometry = geometry
        self._traction_left = traction_left
        self._traction_top = torch.tensor([0.0, 0.0], device=traction_left.device)
        self._traction_hole = torch.tensor([0.0, 0.0], device=traction_left.device)
        self._volume_force = volume_force
        self._num_collocation_points = num_collocation_points
        self._num_points_per_bc = num_points_per_bc
        self._num_samples = len(self._parameters_samples)
        self._samples_collocation: list[TrainingData2DCollocation] = []
        self._samples_stress_bc: list[TrainingData2DStressBC] = []
        self._samples_traction_bc: list[TrainingData2DTractionBC] = []
        self._generate_samples()

    def get_collate_func(self) -> TrainingCollateFunc:
        def collate_func(batch: TrainingBatchList) -> TrainingBatch:
            x_coor_pde_batch = []
            x_params_pde_batch = []
            f_pde_batch = []
            x_coor_stress_bc_batch = []
            x_params_stress_bc_batch = []
            x_coor_traction_bc_batch = []
            x_params_traction_bc_batch = []
            normal_traction_bc_batch = []
            area_frac_traction_bc_batch = []
            y_true_traction_bc_batch = []

            def append_to_pde_batch(sample_pde: TrainingData2DCollocation) -> None:
                x_coor_pde_batch.append(sample_pde.x_coor)
                x_params_pde_batch.append(sample_pde.x_params)
                f_pde_batch.append(sample_pde.f)

            def append_to_stress_bc_batch(
                sample_stress_bc: TrainingData2DStressBC,
            ) -> None:
                x_coor_stress_bc_batch.append(sample_stress_bc.x_coor)
                x_params_stress_bc_batch.append(sample_stress_bc.x_params)

            def append_to_traction_bc_batch(
                sample_traction_bc: TrainingData2DTractionBC,
            ) -> None:
                x_coor_traction_bc_batch.append(sample_traction_bc.x_coor)
                x_params_traction_bc_batch.append(sample_traction_bc.x_params)
                normal_traction_bc_batch.append(sample_traction_bc.normal)
                area_frac_traction_bc_batch.append(sample_traction_bc.area_frac)
                y_true_traction_bc_batch.append(sample_traction_bc.y_true)

            for sample_pde, sample_stress_bc, sample_traction_bc in batch:
                append_to_pde_batch(sample_pde)
                append_to_stress_bc_batch(sample_stress_bc)
                append_to_traction_bc_batch(sample_traction_bc)

            batch_pde = TrainingData2DCollocation(
                x_coor=torch.concat(x_coor_pde_batch, dim=0),
                x_params=torch.concat(x_params_pde_batch, dim=0),
                f=torch.concat(f_pde_batch, dim=0),
            )
            batch_stress_bc = TrainingData2DStressBC(
                x_coor=torch.concat(x_coor_stress_bc_batch, dim=0),
                x_params=torch.concat(x_params_stress_bc_batch, dim=0),
            )
            batch_traction_bc = TrainingData2DTractionBC(
                x_coor=torch.concat(x_coor_traction_bc_batch, dim=0),
                x_params=torch.concat(x_params_traction_bc_batch, dim=0),
                normal=torch.concat(normal_traction_bc_batch),
                area_frac=torch.concat(area_frac_traction_bc_batch),
                y_true=torch.concat(y_true_traction_bc_batch, dim=0),
            )
            return batch_pde, batch_stress_bc, batch_traction_bc

        return collate_func

    def _generate_samples(self) -> None:
        for sample_idx, parameters_sample in enumerate(self._parameters_samples):
            self._add_collocation_sample(parameters_sample)
            self._add_traction_bc_sample(parameters_sample)
            self._add_stress_bc_sample(parameters_sample)

    def _add_collocation_sample(self, parameters_sample: Tensor) -> None:
        shape = (self._num_collocation_points, 1)
        x_coor = self._geometry.create_random_points(self._num_collocation_points)
        x_params = repeat_tensor(parameters_sample, shape)
        f = repeat_tensor(self._volume_force, shape)
        sample = TrainingData2DCollocation(
            x_coor=x_coor.detach(), x_params=x_params.detach(), f=f.detach()
        )
        self._samples_collocation.append(sample)

    def _add_stress_bc_sample(self, parameters_sample: Tensor) -> None:
        x_coor = self._create_coordinates_for_stress_bcs()
        x_params = self._create_parameters_for_bcs(
            parameters_sample, self._num_stress_bcs
        )
        sample = TrainingData2DStressBC(
            x_coor=x_coor.detach(), x_params=x_params.detach()
        )
        self._samples_stress_bc.append(sample)

    def _create_coordinates_for_stress_bcs(self) -> Tensor:
        x_coor_right, _ = self._geometry.create_uniform_points_on_right_boundary(
            self._num_points_per_bc
        )
        x_coor_bottom, _ = self._geometry.create_uniform_points_on_bottom_boundary(
            self._num_points_per_bc
        )
        return torch.concat((x_coor_right, x_coor_bottom), dim=0)

    def _add_traction_bc_sample(self, parameters_sample: Tensor) -> None:
        x_coor, normal = self._create_coordinates_and_normals_for_traction_bcs()
        area_frac = self._calculate_area_fractions_for_traction_bcs()
        x_params = self._create_parameters_for_bcs(
            parameters_sample, self._num_traction_bcs
        )
        y_true = self._create_tractions_for_traction_bcs()
        sample = TrainingData2DTractionBC(
            x_coor=x_coor.detach(),
            x_params=x_params.detach(),
            normal=normal.detach(),
            y_true=y_true.detach(),
            area_frac=area_frac.detach(),
        )
        self._samples_traction_bc.append(sample)

    def _create_coordinates_and_normals_for_traction_bcs(self) -> tuple[Tensor, Tensor]:
        num_points = self._num_points_per_bc
        (
            x_coor_left,
            normal_left,
        ) = self._geometry.create_uniform_points_on_left_boundary(
            num_points, self._bcs_overlap_distance
        )
        (
            x_coor_top,
            normal_top,
        ) = self._geometry.create_uniform_points_on_top_boundary(
            num_points, self._bcs_overlap_distance
        )
        (
            x_coor_hole,
            normal_hole,
        ) = self._geometry.create_uniform_points_on_hole_boundary(
            num_points, self._bcs_overlap_angle_distance
        )

        x_coor = torch.concat((x_coor_left, x_coor_top, x_coor_hole), dim=0)
        normal = torch.concat((normal_left, normal_top, normal_hole), dim=0)
        return x_coor, normal

    def _calculate_area_fractions_for_traction_bcs(self) -> Tensor:
        area_frac_left = self._geometry.calculate_area_fractions_on_left_boundary(
            self._num_points_per_bc
        )
        area_frac_top = self._geometry.calculate_area_fractions_on_top_boundary(
            self._num_points_per_bc
        )
        area_frac_hole = self._geometry.calculate_area_fractions_on_hole_boundary(
            self._num_points_per_bc
        )
        return torch.concat((area_frac_left, area_frac_top, area_frac_hole), dim=0)

    def _create_parameters_for_bcs(
        self, parameters_sample: Tensor, num_bcs: int
    ) -> Tensor:
        shape = (num_bcs * self._num_points_per_bc, 1)
        return repeat_tensor(parameters_sample, shape)

    def _create_tractions_for_traction_bcs(self) -> Tensor:
        shape = (self._num_points_per_bc, 1)
        return torch.concat(
            (
                self._traction_left.repeat(shape),
                self._traction_top.repeat(shape),
                self._traction_hole.repeat(shape),
            ),
            dim=0,
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[
        TrainingData2DCollocation, TrainingData2DStressBC, TrainingData2DTractionBC
    ]:
        sample_collocation = self._samples_collocation[idx]
        sample_stress_bc = self._samples_stress_bc[idx]
        sample_traction_bc = self._samples_traction_bc[idx]
        return sample_collocation, sample_stress_bc, sample_traction_bc
