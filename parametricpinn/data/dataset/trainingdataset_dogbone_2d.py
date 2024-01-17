from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.dataset import (
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
)
from parametricpinn.data.geometry import DogBone2D
from parametricpinn.types import Tensor

TrainingBatch: TypeAlias = tuple[
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
]
TrainingBatchList: TypeAlias = list[TrainingBatch]
TrainingCollateFunc: TypeAlias = Callable[[TrainingBatchList], TrainingBatch]


@dataclass
class DogBoneTrainingDataset2DConfig:
    parameters: Tensor
    traction_right: Tensor
    volume_force: Tensor
    num_collocation_points: int
    num_points_per_bc: int
    bcs_overlap_angle_distance: float


class DogBoneTrainingDataset2D(Dataset):
    def __init__(
        self,
        parameters: Tensor,
        geometry: DogBone2D,
        traction_right: Tensor,
        volume_force: Tensor,
        num_collocation_points: int,
        num_points_per_bc: int,
        bcs_overlap_angle_distance: float,
    ):
        super().__init__()
        self._num_traction_bcs = 8
        self._bcs_overlap_angle_distance = bcs_overlap_angle_distance
        self._bcs_overlap_distance_parallel_left = 0.0
        self._bcs_overlap_distance_parallel_right = 0.0
        self._parameters = parameters
        self._geometry = geometry
        self._traction_right = traction_right
        self._traction_tapered = torch.tensor([0.0, 0.0], device=traction_right.device)
        self._traction_parallel = torch.tensor([0.0, 0.0], device=traction_right.device)
        self._traction_hole = torch.tensor([0.0, 0.0], device=traction_right.device)
        self._volume_force = volume_force
        self._num_collocation_points = num_collocation_points
        self._num_points_per_bc = num_points_per_bc
        self._num_samples = len(self._parameters)
        self._samples_collocation: list[TrainingData2DCollocation] = []
        self._samples_traction_bc: list[TrainingData2DTractionBC] = []
        self._generate_samples()

    def get_collate_func(self) -> TrainingCollateFunc:
        def collate_func(batch: TrainingBatchList) -> TrainingBatch:
            x_coor_pde_batch = []
            x_params_pde_batch = []
            f_pde_batch = []
            x_coor_traction_bc_batch = []
            x_params_traction_bc_batch = []
            normal_traction_bc_batch = []
            area_frac_traction_bc_batch = []
            y_true_traction_bc_batch = []

            def append_to_pde_batch(sample_pde: TrainingData2DCollocation) -> None:
                x_coor_pde_batch.append(sample_pde.x_coor)
                x_params_pde_batch.append(sample_pde.x_params)
                f_pde_batch.append(sample_pde.f)

            def append_to_traction_bc_batch(
                sample_traction_bc: TrainingData2DTractionBC,
            ) -> None:
                x_coor_traction_bc_batch.append(sample_traction_bc.x_coor)
                x_params_traction_bc_batch.append(sample_traction_bc.x_params)
                normal_traction_bc_batch.append(sample_traction_bc.normal)
                area_frac_traction_bc_batch.append(sample_traction_bc.area_frac)
                y_true_traction_bc_batch.append(sample_traction_bc.y_true)

            for sample_pde, sample_traction_bc in batch:
                append_to_pde_batch(sample_pde)
                append_to_traction_bc_batch(sample_traction_bc)

            batch_pde = TrainingData2DCollocation(
                x_coor=torch.concat(x_coor_pde_batch, dim=0),
                x_params=torch.concat(x_params_pde_batch, dim=0),
                f=torch.concat(f_pde_batch, dim=0),
            )
            batch_traction_bc = TrainingData2DTractionBC(
                x_coor=torch.concat(x_coor_traction_bc_batch, dim=0),
                x_params=torch.concat(x_params_traction_bc_batch, dim=0),
                normal=torch.concat(normal_traction_bc_batch),
                area_frac=torch.concat(area_frac_traction_bc_batch),
                y_true=torch.concat(y_true_traction_bc_batch, dim=0),
            )
            return batch_pde, batch_traction_bc

        return collate_func

    def _generate_samples(self) -> None:
        for sample_idx, parameters_sample in enumerate(self._parameters):
            self._add_collocation_sample(parameters_sample)
            self._add_traction_bc_sample(parameters_sample)
            print(f"Add training sample {sample_idx + 1} / {self._num_samples}")

    def _add_collocation_sample(self, parameters_sample: Tensor) -> None:
        shape = (self._num_collocation_points, 1)
        x_coor = self._geometry.create_random_points(self._num_collocation_points)
        x_params = repeat_tensor(parameters_sample, shape)
        f = repeat_tensor(self._volume_force, shape)
        sample = TrainingData2DCollocation(
            x_coor=x_coor.detach(), x_params=x_params.detach(), f=f.detach()
        )
        self._samples_collocation.append(sample)

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
        # right
        num_points = self._num_points_per_bc
        (
            x_coor_right,
            normal_right,
        ) = self._geometry.create_uniform_points_on_right_tapered_boundary(num_points)
        # top
        (
            x_coor_top_left_tapered,
            normal_top_left_tapered,
        ) = self._geometry.create_uniform_points_on_top_left_tapered_boundary(
            num_points, self._bcs_overlap_angle_distance
        )
        (
            x_coor_top_right_tapered,
            normal_top_right_tapered,
        ) = self._geometry.create_uniform_points_on_top_right_tapered_boundary(
            num_points, self._bcs_overlap_angle_distance
        )
        (
            x_coor_top_parallel_complete,
            normal_top_parallel_complete,
        ) = self._geometry.create_uniform_points_on_top_parallel_boundary(
            num_points + 2,
            self._bcs_overlap_distance_parallel_left,
            self._bcs_overlap_distance_parallel_right,
        )
        x_coor_top_parallel = x_coor_top_parallel_complete[1:-1, :]
        normal_top_parallel = normal_top_parallel_complete[1:-1, :]
        x_coor_top = torch.concat(
            (x_coor_top_left_tapered, x_coor_top_parallel, x_coor_top_right_tapered),
            dim=0,
        )
        normal_top = torch.concat(
            (normal_top_left_tapered, normal_top_parallel, normal_top_right_tapered),
            dim=0,
        )
        # bottom
        (
            x_coor_bottom_left_tapered,
            normal_bottom_left_tapered,
        ) = self._geometry.create_uniform_points_on_bottom_left_tapered_boundary(
            num_points, self._bcs_overlap_angle_distance
        )
        (
            x_coor_bottom_right_tapered,
            normal_bottom_right_tapered,
        ) = self._geometry.create_uniform_points_on_bottom_right_tapered_boundary(
            num_points, self._bcs_overlap_angle_distance
        )
        (
            x_coor_bottom_parallel_complete,
            normal_bottom_parallel_complete,
        ) = self._geometry.create_uniform_points_on_bottom_parallel_boundary(
            num_points + 2,
            self._bcs_overlap_distance_parallel_left,
            self._bcs_overlap_distance_parallel_right,
        )
        x_coor_bottom_parallel = x_coor_bottom_parallel_complete[1:-1, :]
        normal_bottom_parallel = normal_bottom_parallel_complete[1:-1, :]
        x_coor_bottom = torch.concat(
            (
                x_coor_bottom_left_tapered,
                x_coor_bottom_parallel,
                x_coor_bottom_right_tapered,
            ),
            dim=0,
        )
        normal_bottom = torch.concat(
            (
                normal_bottom_left_tapered,
                normal_bottom_parallel,
                normal_bottom_right_tapered,
            ),
            dim=0,
        )
        # hole
        (
            x_coor_hole,
            normal_hole,
        ) = self._geometry.create_uniform_points_on_hole_boundary(num_points)

        x_coor = torch.concat(
            (x_coor_right, x_coor_top, x_coor_bottom, x_coor_hole), dim=0
        )
        normal = torch.concat(
            (normal_right, normal_top, normal_bottom, normal_hole), dim=0
        )
        return x_coor, normal

    def _calculate_area_fractions_for_traction_bcs(self) -> Tensor:
        num_points = self._num_points_per_bc
        # right
        area_frac_right = (
            self._geometry.calculate_area_fractions_on_vertical_tapered_boundary(
                num_points
            )
        )
        # top and bottom
        area_frac_tapered = self._geometry.calculate_area_fraction_on_tapered_boundary(
            num_points
        )
        area_frac_parallel = (
            self._geometry.calculate_area_fractions_on_horizontal_parallel_boundary(
                num_points
            )
        )
        area_frac_top = area_frac_bottom = torch.concat(
            (area_frac_tapered, area_frac_parallel, area_frac_tapered), dim=0
        )
        # hole
        area_frac_hole = self._geometry.calculate_area_fractions_on_hole_boundary(
            num_points
        )
        return torch.concat(
            (area_frac_right, area_frac_top, area_frac_bottom, area_frac_hole), dim=0
        )

    def _create_parameters_for_bcs(
        self, parameters_sample: Tensor, num_bcs: int
    ) -> Tensor:
        shape = (num_bcs * self._num_points_per_bc, 1)
        return repeat_tensor(parameters_sample, shape)

    def _create_tractions_for_traction_bcs(self) -> Tensor:
        shape = (self._num_points_per_bc, 1)
        traction_top = traction_bottom = torch.concat(
            (
                self._traction_tapered.repeat(shape),
                self._traction_parallel.repeat(shape),
                self._traction_tapered.repeat(shape),
            ),
            dim=0,
        )
        return torch.concat(
            (
                self._traction_right.repeat(shape),
                traction_top,
                traction_bottom,
                self._traction_hole.repeat(shape),
            ),
            dim=0,
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[TrainingData2DCollocation, TrainingData2DTractionBC]:
        sample_collocation = self._samples_collocation[idx]
        sample_traction_bc = self._samples_traction_bc[idx]
        return sample_collocation, sample_traction_bc
