import math

import shapely
import torch

from calibrationpinn.types import ShapelyPolygon, Tensor


class QuarterPlateWithHole2D:
    def __init__(self, edge_length: float, radius: float) -> None:
        self.edge_length = edge_length
        self.radius = radius
        self._x_min = -edge_length
        self._x_max = 0.0
        self._y_min = 0.0
        self._y_max = edge_length
        self._x_center = 0.0
        self._y_center = 0.0
        self._angle_min = 0.0
        self._angle_max = 90.0
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

    def create_uniform_points_on_left_boundary(
        self, num_points: int, bcs_overlap_distance: float
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        y_min = self._y_min + bcs_overlap_distance
        y_max = self._y_max
        coordinates_x = torch.full(shape, self._x_min, requires_grad=True)
        coordinates_y = torch.linspace(
            y_min, y_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([-1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_top_boundary(
        self, num_points: int, bcs_overlap_distance: float
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        x_min = self._x_min + bcs_overlap_distance
        x_max = self._x_max - bcs_overlap_distance
        coordinates_x = torch.linspace(
            x_min, x_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates_y = torch.full(shape, self._y_max, requires_grad=True)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, 1.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_right_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.full(shape, self._x_max, requires_grad=True)
        coordinates_y = torch.linspace(
            self._y_min + self.radius, self._y_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_hole_boundary(
        self, num_points: int, bcs_overlap_angle_distance: float
    ) -> tuple[Tensor, Tensor]:
        min_angle = self._angle_min + bcs_overlap_angle_distance
        max_angle = self._angle_max - bcs_overlap_angle_distance
        angles = torch.linspace(min_angle, max_angle, num_points).view(num_points, 1)
        delta_x = torch.cos(torch.deg2rad(angles)) * self.radius
        delta_y = torch.sin(torch.deg2rad(angles)) * self.radius
        coordinates_x = self._x_center - delta_x
        coordinates_y = self._y_center + delta_y
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = delta_x
        normals_y = -delta_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.radius
        return coordinates, normals

    def create_uniform_points_on_bottom_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.linspace(
            self._x_min, self._x_max - self.radius, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates_y = torch.full(shape, self._y_min, requires_grad=True)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, -1.0]).repeat(shape)
        return coordinates, normals

    def calculate_area_fractions_on_left_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.edge_length / num_points]).repeat(shape)

    def calculate_area_fractions_on_top_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.edge_length / num_points]).repeat(shape)

    def calculate_area_fractions_on_hole_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        edge_length_hole = 1 / 4 * 2.0 * math.pi * self.radius
        return torch.tensor([edge_length_hole / num_points]).repeat(shape)

    def _create_one_random_point(self) -> Tensor:
        min_coordinates = torch.tensor([self._x_min, self._y_min])
        normalized_delta = self._sobol_engine.draw()[0]
        delta = normalized_delta * torch.tensor([self.edge_length, self.edge_length])
        return min_coordinates + delta

    def _is_point_in_shape(self, point: Tensor) -> bool:
        _point = point.detach().numpy()
        return self._shape.contains(shapely.Point(_point[0], _point[1]))

    def _create_shape(self) -> ShapelyPolygon:
        plate = shapely.box(self._x_min, self._y_min, self._x_max, self._y_max)
        hole = shapely.Point(0, 0).buffer(self.radius)
        plate_with_hole = shapely.difference(plate, hole)
        return plate_with_hole
