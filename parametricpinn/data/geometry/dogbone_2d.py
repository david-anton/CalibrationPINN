import math

import shapely
import torch

from parametricpinn.types import ShapelyPolygon, Tensor


class DogBone2D:
    def __init__(self) -> None:
        self._origin_x = 0
        self._origin_y = 0
        self._box_length = 120
        self._box_height = 30
        self._half_box_length = self._box_length / 2
        self._half_box_height = self._box_height / 2
        self._parallel_length = 90
        self._parallel_height = 20
        self._half_parallel_length = self._parallel_length / 2
        self._half_parallel_height = self._parallel_height / 2
        self._cut_parallel_height = (self._box_height - self._parallel_height) / 2
        self._tapered_radius = 25
        self._plate_hole_radius = 4
        self._angle_min_tapered = 0
        self._angle_max_tapered = math.degrees(math.atan((self._half_box_height - self._half_parallel_height) / self._tapered_radius))
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
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.linspace(
            -self._half_parallel_length, self._half_parallel_length, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates_y = torch.full(shape, self._half_parallel_height, requires_grad=True)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, 1.0]).repeat(shape)
        return coordinates, normals
    
    def create_uniform_points_on_top_left_tapered_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        abs_rad_component_x, abs_rad_component_y = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(num_points)
        coordinates_x = -self._half_parallel_length - abs_rad_component_x
        coordinates_y = self._half_parallel_height + (self._tapered_radius - abs_rad_component_y)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = abs_rad_component_x
        normals_y = abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self._tapered_radius
        return coordinates, normals
    
    def create_uniform_points_on_top_right_tapered_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        abs_rad_component_x, abs_rad_component_y = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(num_points)
        coordinates_x = self._half_parallel_length + abs_rad_component_x
        coordinates_y = self._half_parallel_height + (self._tapered_radius - abs_rad_component_y)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = -abs_rad_component_x
        normals_y = abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self._tapered_radius
        return coordinates, normals

    def create_uniform_points_on_bottom_parallel_boundary(
            self, num_points: int
        ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.linspace(
            -self._half_parallel_length, self._half_parallel_length, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates_y = torch.full(shape, -self._half_parallel_height, requires_grad=True)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([0.0, -1.0]).repeat(shape)
        return coordinates, normals
    
    def create_uniform_points_on_bottom_left_tapered_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        abs_rad_component_x, abs_rad_component_y = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(num_points)
        coordinates_x = -self._half_parallel_length - abs_rad_component_x
        coordinates_y = -self._half_parallel_height - (self._tapered_radius - abs_rad_component_y)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = abs_rad_component_x
        normals_y = -abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self._tapered_radius
        return coordinates, normals
    
    def create_uniform_points_on_bottom_right_tapered_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        abs_rad_component_x, abs_rad_component_y = self._calculate_absolute_radial_coordinate_components_for_tapered_boundaries(num_points)
        coordinates_x = self._half_parallel_length + abs_rad_component_x
        coordinates_y = -self._half_parallel_height - (self._tapered_radius - abs_rad_component_y)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = -abs_rad_component_x
        normals_y = -abs_rad_component_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self._tapered_radius
        return coordinates, normals

    def create_uniform_points_on_right_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        shape = (num_points, 1)
        coordinates_x = torch.full(shape, self._half_box_length, requires_grad=True)
        coordinates_y = torch.linspace(
            -self._half_box_height, self._half_box_height, num_points, requires_grad=True
        ).view(num_points, 1)
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals = torch.tensor([1.0, 0.0]).repeat(shape)
        return coordinates, normals

    def create_uniform_points_on_hole_boundary(
        self, num_points: int
    ) -> tuple[Tensor, Tensor]:
        extended_num_points = num_points + 1
        angles = torch.linspace(
            0, 360, extended_num_points
        ).view(extended_num_points, 1)[:-1, :]
        delta_x = torch.cos(torch.deg2rad(angles)) * self._tapered_radius
        delta_y = torch.sin(torch.deg2rad(angles)) * self._tapered_radius
        coordinates_x = self._origin_x - delta_x
        coordinates_y = self._origin_y + delta_y
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1)
        normals_x = delta_x
        normals_y = -delta_y
        normals = torch.concat((normals_x, normals_y), dim=1) / self.radius
        return coordinates, normals

    def calculate_area_fractions_on_horizontal_parallel_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self._parallel_length / num_points]).repeat(shape)

    def calculate_area_fractions_on_vertical_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        return torch.tensor([self._box_height / num_points]).repeat(shape)

    def calculate_area_fraction_on_tapered_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        angle = self._angle_max_tapered - self._angle_min_tapered
        full_perimeter = 2.0 * math.pi * self._tapered_radius
        edge_length = (angle/360) * full_perimeter
        return torch.tensor([edge_length / num_points]).repeat(shape)

    def calculate_area_fractions_on_hole_boundary(self, num_points) -> Tensor:
        shape = (num_points, 1)
        edge_length = 2.0 * math.pi * self._plate_hole_radius
        return torch.tensor([edge_length / num_points]).repeat(shape)

    def _create_one_random_point(self) -> Tensor:
        coordinate_x = self._create_one_random_coordinate(-self._half_box_length, self._half_box_length)
        coordinate_y = self._create_one_random_coordinate(-self._half_box_height, self._half_box_height)
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
        box = shapely.box(
            -self._half_box_length,
            -self._half_parallel_height,
            self._box_length,
            self._box_height,
        )
        cut_parallel_top = shapely.box(
            -self._half_parallel_length,
            self._half_parallel_height,
            self._parallel_length,
            self._cut_parallel_height,
        )
        cut_parallel_bottom = shapely.box(
            -self._half_parallel_length,
            -self._half_box_height,
            self._parallel_length,
            self._cut_parallel_height,
        )
        cut_tapered_top_left = shapely.Point(
            -self._half_parallel_length,
            self._half_parallel_height + self._tapered_radius,
        ).buffer(self._tapered_radius)
        cut_tapered_top_right = shapely.Point(
            self._half_parallel_length,
            self._half_parallel_height + self._tapered_radius,
        ).buffer(self._tapered_radius)
        cut_tapered_bottom_left = shapely.Point(
            -self._half_parallel_length,
            -self._half_parallel_height - self._tapered_radius,
        ).buffer(self._tapered_radius)
        cut_tapered_bottom_right = shapely.Point(
            self._half_parallel_length,
            -self._half_parallel_height - self._tapered_radius,
        ).buffer(self._tapered_radius)
        plate_hole = shapely.Point(
            self._origin_x,
            self._origin_y,
        ).buffer(self._plate_hole_radius)
        dog_bone = shapely.difference(
            box,
            [
                cut_parallel_top,
                cut_parallel_bottom,
                cut_tapered_top_left,
                cut_tapered_top_right,
                cut_tapered_bottom_left,
                cut_tapered_bottom_right,
                plate_hole
            ],
        )
        return dog_bone

    def _calculate_absolute_radial_coordinate_components_for_tapered_boundaries(self, num_points: int) -> list[Tensor, Tensor]:
        angles = torch.linspace(
            self._angle_min_tapered, self._angle_max_tapered
        ).view(num_points, 1)
        abs_components_x = torch.sin(torch.deg2rad(angles)) * self._tapered_radius
        abs_components_y = torch.cos(torch.deg2rad(angles)) * self._tapered_radius
        return abs_components_x, abs_components_y