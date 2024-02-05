import shapely
import torch

from parametricpinn.types import ShapelyPolygon, Tensor


class Plate2D:
    def __init__(self, plate_length: float, plate_height: float) -> None:
        self.length = plate_length
        self._half_length = plate_length / 2
        self.height = plate_height
        self._half_height = plate_height / 2
        self._x_min = -self._half_length
        self._x_max = self._half_length
        self._y_min = -self._half_height
        self._y_max = self._half_height
        self._x_center = 0.0
        self._y_center = 0.0
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
            self._y_min, self._y_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_bottom_boundary(
        self, num_points: int, bcs_overlap_distance: float
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        x_min = self._x_min + bcs_overlap_distance
        x_max = self._x_max - bcs_overlap_distance
        coordinates_x = torch.linspace(
            x_min, x_max, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates_y = torch.full(shape, self._y_min, requires_grad=True)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, -1.0]).repeat(shape)
        return coordinates, normals

    def calculate_area_fractions_on_horizontal_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.length / num_points]).repeat(shape)

    def calculate_area_fractions_on_vertical_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self.height / num_points]).repeat(shape)

    def _create_one_random_point(self) -> Tensor:
        min_coordinates = torch.tensor([self._x_min, self._y_min])
        normalized_delta = self._sobol_engine.draw()[0]
        delta = normalized_delta * torch.tensor([self.length, self.height])
        return min_coordinates + delta

    def _is_point_in_shape(self, point: Tensor) -> bool:
        _point = point.detach().numpy()
        return self._shape.contains(shapely.Point(_point[0], _point[1]))

    def _create_shape(self) -> ShapelyPolygon:
        plate = shapely.box(self._x_min, self._y_min, self._x_max, self._y_max)
        return plate
