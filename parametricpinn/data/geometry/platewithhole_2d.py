import math

import shapely
import torch

from parametricpinn.types import ShapelyPolygon, Tensor


class PlateWithHole2D:
    def __init__(
        self, plate_length: float, plate_height: float, hole_radius: float
    ) -> None:
        self.length = plate_length
        self.height = plate_height
        self.radius = hole_radius
        self._x_min = 0.0
        self._x_max = plate_length
        self._y_min = 0.0
        self._y_max = plate_height
        self._x_center = plate_length / 2
        self._y_center = plate_height / 2
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
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.full(shape, self._x_min, requires_grad=True)
        coordinates_y = torch.linspace(
            self._y_min, self._y_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([-1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_top_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.linspace(
            self._x_min, self._x_max, num_points, requires_grad=True
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
            self._y_min, self._y_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_bottom_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.linspace(
            self._x_min, self._x_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates_y = torch.full(shape, self._y_min, requires_grad=True)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, -1.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_hole_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        extended_num_points = num_points + 1
        angles = torch.linspace(
            0, 360, extended_num_points
        ).view(extended_num_points, 1)[:-1, :]
        delta_x = torch.cos(torch.deg2rad(angles)) * self.radius
        delta_y = torch.sin(torch.deg2rad(angles)) * self.radius
        coordinates_x = self._x_center - delta_x
        coordinates_y = self._y_center + delta_y
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = delta_x
        normals_y = -delta_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.radius
        return coordinates, normals

    def calculate_area_fractions_on_horizontal_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.length / num_points]).repeat(shape)

    def calculate_area_fractions_on_vertical_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.height / num_points]).repeat(shape)

    def calculate_area_fractions_on_hole_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        edge_length_hole = 2.0 * math.pi * self.radius
        return torch.tensor([edge_length_hole / num_points]).repeat(shape)

    def _create_one_random_point(self) -> Tensor:
        coordinate_x = self._create_one_random_coordinate(self._x_min, self._x_max)
        coordinate_y = self._create_one_random_coordinate(self._y_min, self._y_max)
        return torch.concat((coordinate_x, coordinate_y))

    def _create_one_random_coordinate(
        self, min_coordinate: float, max_coordinate: float
    ) -> Tensor:
        return min_coordinate + torch.rand((1), requires_grad=True) * (
            max_coordinate - min_coordinate
        )

    def _is_point_in_shape(self, point: Tensor) -> bool:
        _point = point.detach().numpy()
        return self._shape.contains(shapely.Point(_point[0], _point[1]))

    def _create_shape(self) -> ShapelyPolygon:
        plate = shapely.box(self._x_min, self._y_min, self._x_max, self._y_max)
        hole = shapely.Point(self._x_center, self._y_center).buffer(self.radius)
        plate_with_hole = shapely.difference(plate, hole)
        return plate_with_hole
