import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import shapely
import torch

from parametricpinn.types import ShapelyPolygon, Tensor


class DogBone2DBase(ABC):
    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        left_half_box_length: float,
        right_half_box_length: float,
        box_length: float,
        box_height: float,
        half_box_height: float,
        left_half_parallel_length: float,
        right_half_parallel_length: float,
        parallel_length: float,
        parallel_height: float,
        half_parallel_height: float,
        cut_parallel_height: float,
        tapered_radius: float,
        plate_hole_radius: float,
        angle_max_tapered: float,
    ) -> None:
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.left_half_box_length = left_half_box_length
        self.right_half_box_length = right_half_box_length
        self.box_length = box_length
        self.box_height = box_height
        self.half_box_height = half_box_height
        self.left_half_parallel_length = left_half_parallel_length
        self.right_half_parallel_length = right_half_parallel_length
        self.parallel_length = parallel_length
        self.parallel_height = parallel_height
        self.half_parallel_height = half_parallel_height
        self.cut_parallel_height = cut_parallel_height
        self.tapered_radius = tapered_radius
        self.plate_hole_radius = plate_hole_radius
        self.angle_min_tapered = 0
        self.angle_max_tapered = angle_max_tapered
        self._sobol_engine = torch.quasirandom.SobolEngine(dimension=2)
        self._shape = self._create_shape()

    def create_random_points(self, num_points: int) -> Tensor:
        point_count = 0
        point_list = []
        while point_count < num_points:
            point = self._create_one_random_point()
            if self._is_point_in_shape(point):
                point_list.append(point)
                point_count += 1
        return torch.vstack(point_list)

    def create_uniform_points_on_top_parallel_boundary(
        self,
        num_points: int,
        bcs_overlap_distance_left: float,
        bcs_overlap_distance_right: float,
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.linspace(
            -self.left_half_parallel_length + bcs_overlap_distance_left,
            self.right_half_parallel_length - bcs_overlap_distance_right,
            num_points,
            requires_grad=True,
        ).view(num_points, 1)
        coordinates_y = torch.full(shape, self.half_parallel_height, requires_grad=True)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, 1.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_top_left_tapered_boundary(
        self, num_points: int, overlap_angle_distance: float
    ) -> tuple[Tensor, Tensor]:
        (
            abs_rad_component_x,
            abs_rad_component_y,
        ) = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(
            num_points, overlap_angle_distance
        )
        coordinates_x = -self.left_half_parallel_length - abs_rad_component_x
        coordinates_y = self.half_parallel_height + (
            self.tapered_radius - abs_rad_component_y
        )
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = abs_rad_component_x
        normals_y = abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.tapered_radius
        return self._reverse_order_of_data_points(
            coordinates
        ), self._reverse_order_of_data_points(normals)

    def create_uniform_points_on_bottom_parallel_boundary(
        self,
        num_points: int,
        bcs_overlap_distance_left: float,
        bcs_overlap_distance_right: float,
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.linspace(
            -self.left_half_parallel_length + bcs_overlap_distance_left,
            self.right_half_parallel_length - bcs_overlap_distance_right,
            num_points,
            requires_grad=True,
        ).view(num_points, 1)
        coordinates_y = torch.full(
            shape, -self.half_parallel_height, requires_grad=True
        )
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, -1.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_bottom_left_tapered_boundary(
        self, num_points: int, overlap_angle_distance: float
    ) -> tuple[Tensor, Tensor]:
        (
            abs_rad_component_x,
            abs_rad_component_y,
        ) = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(
            num_points, overlap_angle_distance
        )
        coordinates_x = -self.left_half_parallel_length - abs_rad_component_x
        coordinates_y = -self.half_parallel_height - (
            self.tapered_radius - abs_rad_component_y
        )
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = abs_rad_component_x
        normals_y = -abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.tapered_radius
        return self._reverse_order_of_data_points(
            coordinates
        ), self._reverse_order_of_data_points(normals)

    def create_uniform_points_on_hole_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        extended_num_points = num_points + 1
        angles = torch.linspace(0, 360, extended_num_points).view(
            extended_num_points, 1
        )[:-1, :]
        delta_x = torch.cos(torch.deg2rad(angles)) * self.plate_hole_radius
        delta_y = torch.sin(torch.deg2rad(angles)) * self.plate_hole_radius
        coordinates_x = self.origin_x - delta_x
        coordinates_y = self.origin_y + delta_y
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = delta_x
        normals_y = -delta_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.plate_hole_radius
        return coordinates, normals

    def calculate_area_fractions_on_horizontal_parallel_boundary(
        self, num_points
    ) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor(self.parallel_length / num_points).repeat(shape)

    def calculate_area_fraction_on_tapered_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        angle = self.angle_max_tapered - self.angle_min_tapered
        full_perimeter = 2.0 * math.pi * self.tapered_radius
        edge_length = (angle / 360) * full_perimeter
        return torch.tensor([edge_length / num_points]).repeat(shape)

    def calculate_area_fractions_on_hole_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        edge_length = 2.0 * math.pi * self.plate_hole_radius
        return torch.tensor([edge_length / num_points]).repeat(shape)

    def _create_one_random_point(self) -> Tensor:
        min_coordinates = torch.tensor(
            [-self.left_half_box_length, -self.half_box_height]
        )
        normalized_delta = self._sobol_engine.draw()[0]
        delta = normalized_delta * torch.tensor(
            [
                self.box_length,
                self.box_height,
            ]
        )
        return min_coordinates + delta

    def _is_point_in_shape(self, point: Tensor) -> bool:
        _point = point.detach().numpy()
        return self._shape.contains(shapely.Point(_point[0], _point[1]))

    def _calculate_absolute_radial_coordinate_components_for_tapered_boundaries(
        self, num_points: int, overlap_angle_distance: float
    ) -> list[Tensor, Tensor]:
        min_angle = self.angle_min_tapered
        max_angle = self.angle_max_tapered - overlap_angle_distance
        angles = torch.linspace(min_angle, max_angle, num_points).view(num_points, 1)
        abs_components_x = torch.sin(torch.deg2rad(angles)) * self.tapered_radius
        abs_components_y = torch.cos(torch.deg2rad(angles)) * self.tapered_radius
        return abs_components_x, abs_components_y

    def _reverse_order_of_data_points(self, data_tensors: Tensor) -> Tensor:
        return torch.flip(data_tensors, dims=[0])

    @abstractmethod
    def _create_shape(self) -> ShapelyPolygon:
        pass


@dataclass
class DogBoneGeometryConfig:
    origin_x = 0.0
    origin_y = 0.0
    box_length = 120.0
    box_height = 30.0
    half_box_length = box_length / 2
    half_box_height = box_height / 2
    parallel_length = 90.0
    parallel_height = 20.0
    half_parallel_length = parallel_length / 2
    half_parallel_height = parallel_height / 2
    cut_parallel_height = half_box_height - half_parallel_height
    tapered_radius = 25.0
    plate_hole_radius = 4.0
    angle_max_tapered = math.degrees(
        math.asin((half_box_length - half_parallel_length) / tapered_radius)
    )


class DogBone2D(DogBone2DBase):
    def __init__(self, geometry_config: DogBoneGeometryConfig) -> None:
        super().__init__(
            origin_x=geometry_config.origin_x,
            origin_y=geometry_config.origin_y,
            left_half_box_length=geometry_config.half_box_length,
            right_half_box_length=geometry_config.half_box_length,
            box_length=geometry_config.box_length,
            box_height=geometry_config.box_height,
            half_box_height=geometry_config.half_box_height,
            left_half_parallel_length=geometry_config.half_parallel_length,
            right_half_parallel_length=geometry_config.half_parallel_length,
            parallel_length=geometry_config.parallel_length,
            parallel_height=geometry_config.parallel_height,
            half_parallel_height=geometry_config.half_parallel_height,
            cut_parallel_height=geometry_config.cut_parallel_height,
            tapered_radius=geometry_config.tapered_radius,
            plate_hole_radius=geometry_config.plate_hole_radius,
            angle_max_tapered=geometry_config.angle_max_tapered,
        )
        self._shape = self._create_shape()

    def create_uniform_points_on_top_right_tapered_boundary(
        self, num_points: int, overlap_angle_distance: float
    ) -> tuple[Tensor, Tensor]:
        (
            abs_rad_component_x,
            abs_rad_component_y,
        ) = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(
            num_points, overlap_angle_distance
        )
        coordinates_x = self.right_half_parallel_length + abs_rad_component_x
        coordinates_y = self.half_parallel_height + (
            self.tapered_radius - abs_rad_component_y
        )
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = -abs_rad_component_x
        normals_y = abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.tapered_radius
        return coordinates, normals

    def create_uniform_points_on_bottom_right_tapered_boundary(
        self, num_points: int, overlap_angle_distance: float
    ) -> tuple[Tensor, Tensor]:
        (
            abs_rad_component_x,
            abs_rad_component_y,
        ) = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(
            num_points, overlap_angle_distance
        )
        coordinates_x = self.right_half_parallel_length + abs_rad_component_x
        coordinates_y = -self.half_parallel_height - (
            self.tapered_radius - abs_rad_component_y
        )
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = -abs_rad_component_x
        normals_y = -abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.tapered_radius
        return coordinates, normals

    def create_uniform_points_on_right_tapered_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.full(
            shape, self.right_half_box_length, requires_grad=True
        )
        coordinates_y = torch.linspace(
            -self.half_box_height,
            self.half_box_height,
            num_points,
            requires_grad=True,
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def calculate_area_fractions_on_vertical_tapered_boundary(
        self, num_points
    ) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.box_height / num_points]).repeat(shape)

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
        cut_tapered_top_right = shapely.Point(
            self.right_half_parallel_length,
            self.half_parallel_height + self.tapered_radius,
        ).buffer(self.tapered_radius)
        cut_tapered_bottom_left = shapely.Point(
            -self.left_half_parallel_length,
            -self.half_parallel_height - self.tapered_radius,
        ).buffer(self.tapered_radius)
        cut_tapered_bottom_right = shapely.Point(
            self.right_half_parallel_length,
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
            - cut_tapered_top_right
            - cut_tapered_bottom_left
            - cut_tapered_bottom_right
            - plate_hole
        )
        return dog_bone
