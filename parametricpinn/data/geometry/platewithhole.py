import torch
from shapely.geometry import Point, Polygon, box

from parametricpinn.types import Tensor


class PlateWithHole:
    def __init__(self, edge_length: float, radius: float) -> None:
        self.edge_length = edge_length
        self.radius = radius
        self._x_min = -self.edge_length
        self._x_max = 0
        self._y_min = 0
        self._y_max = self.edge_length
        self._angle_min = 0
        self._angle_max = 90
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

    def create_uniform_points_on_hole_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        angles = torch.linspace(self._angle_min, self._angle_max, num_points).view(
            num_points, 1
        )
        coordinates_x = -torch.cos(torch.deg2rad(angles)) * self.radius
        coordinates_y = torch.sin(torch.deg2rad(angles)) * self.radius
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = -coordinates_x
        normals_y = -coordinates_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.radius
        return coordinates, normals

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
        return self._shape.contains(Point(_point[0], _point[1]))

    def _create_shape(self) -> Polygon:
        plate = box(self._x_min, self._x_max, self._y_min, self._y_max)
        hole = Point(0, 0).buffer(self.radius)
        plate_with_hole = plate.difference(hole)
        return plate_with_hole
