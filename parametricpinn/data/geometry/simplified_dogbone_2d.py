import math
from dataclasses import dataclass

import shapely
import torch

from parametricpinn.data.geometry.dogbone_2d import DogBone2DBase
from parametricpinn.types import ShapelyPolygon, Tensor


@dataclass
class SimplifiedDogBoneGeometryConfig:
    origin_x = 0.0
    origin_y = 0.0
    left_half_box_length = 60.0
    right_half_box_length = 40.0
    box_length = left_half_box_length + right_half_box_length
    box_height = 30.0
    half_box_height = box_height / 2
    left_half_parallel_length = 45.0
    right_half_parallel_length = 40.0
    parallel_length = left_half_parallel_length + right_half_parallel_length
    parallel_height = 20.0
    half_parallel_height = parallel_height / 2
    cut_parallel_height = half_box_height - half_parallel_height
    tapered_radius = 25.0
    plate_hole_radius = 4.0
    angle_max_tapered = math.degrees(
        math.asin((box_length - parallel_length) / tapered_radius)
    )


class SimplifiedDogBone2D(DogBone2DBase):
    def __init__(self, geometry_config: SimplifiedDogBoneGeometryConfig) -> None:
        super().__init__(
            origin_x=geometry_config.origin_x,
            origin_y=geometry_config.origin_y,
            left_half_box_length=geometry_config.left_half_box_length,
            right_half_box_length=geometry_config.right_half_box_length,
            box_length=geometry_config.box_length,
            box_height=geometry_config.box_height,
            half_box_height=geometry_config.half_box_height,
            left_half_parallel_length=geometry_config.left_half_parallel_length,
            right_half_parallel_length=geometry_config.right_half_parallel_length,
            parallel_length=geometry_config.parallel_length,
            parallel_height=geometry_config.parallel_height,
            half_parallel_height=geometry_config.half_parallel_height,
            cut_parallel_height=geometry_config.cut_parallel_height,
            tapered_radius=geometry_config.tapered_radius,
            plate_hole_radius=geometry_config.plate_hole_radius,
            angle_max_tapered=geometry_config.angle_max_tapered,
        )
        self._shape = self._create_shape()

    def create_uniform_points_on_right_parallel_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.full(
            shape, self.right_half_parallel_length, requires_grad=True
        )
        coordinates_y = torch.linspace(
            -self.half_parallel_height,
            self.half_parallel_height,
            num_points,
            requires_grad=True,
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def calculate_area_fractions_on_vertical_parallel_boundary(
        self, num_points
    ) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.parallel_height / num_points]).repeat(shape)

    def _create_shape(self) -> ShapelyPolygon:
        box = shapely.box(
            -self.left_half_box_length,
            -self.half_box_height,
            self.right_half_box_length,
            self.half_box_height,
        )
        cut_parallel_top = shapely.box(
            -self.left_half_parallel_length,
            self.half_parallel_height,
            self.right_half_parallel_length,
            self.half_box_height,
        )
        cut_parallel_bottom = shapely.box(
            -self.left_half_parallel_length,
            -self.half_box_height,
            self.right_half_parallel_length,
            -self.half_parallel_height,
        )
        cut_tapered_top_left = shapely.Point(
            -self.left_half_parallel_length,
            self.half_parallel_height + self.tapered_radius,
        ).buffer(self.tapered_radius)
        cut_tapered_bottom_left = shapely.Point(
            -self.left_half_parallel_length,
            -self.half_parallel_height - self.tapered_radius,
        ).buffer(self.tapered_radius)
        plate_hole = shapely.Point(
            self.origin_x,
            self.origin_y,
        ).buffer(self.plate_hole_radius)
        dog_bone = (
            box
            - cut_parallel_top
            - cut_parallel_bottom
            - cut_tapered_top_left
            - cut_tapered_bottom_left
            - plate_hole
        )
        return dog_bone
