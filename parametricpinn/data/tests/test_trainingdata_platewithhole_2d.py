import math

import pytest
import shapely
import torch

from parametricpinn.data.dataset import (
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
)
from parametricpinn.data.trainingdata_2d import (
    PlateWithHoleTrainingDataset2D,
    PlateWithHoleTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.settings import set_default_dtype
from parametricpinn.types import ShapelyPolygon, Tensor

plate_length = 20.0
plate_height = 10.0
hole_radius = 2.0
x_min = -plate_length / 2
x_max = plate_length / 2
y_min = -plate_height / 2
y_max = plate_height / 2
origin_x = 0.0
origin_y = 0.0
parameters = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
volume_foce = torch.tensor([10.0, 10.0], dtype=torch.float64)
traction_right = torch.tensor([-100.0, 0.0], dtype=torch.float64)
traction_top = torch.tensor([0.0, 0.0], dtype=torch.float64)
traction_bottom = torch.tensor([0.0, 0.0], dtype=torch.float64)
traction_hole = torch.tensor([0.0, 0.0], dtype=torch.float64)
num_samples = len(parameters)
num_collocation_points = 32
num_points_per_bc = 3
num_traction_bcs = 4
num_points_traction_bcs = num_traction_bcs * num_points_per_bc
bcs_overlap_distance = 1.0

set_default_dtype(torch.float64)


### Test TrainingDataset
def _create_plate_with_hole_shape() -> ShapelyPolygon:
    plate = shapely.box(x_min, y_min, x_max, y_max)
    hole = shapely.Point(origin_x, origin_y).buffer(hole_radius)
    return shapely.difference(plate, hole)


shape = _create_plate_with_hole_shape()


@pytest.fixture
def sut() -> PlateWithHoleTrainingDataset2D:
    config = PlateWithHoleTrainingDataset2DConfig(
        parameters=parameters,
        plate_length=plate_length,
        plate_height=plate_height,
        hole_radius=hole_radius,
        traction_right=traction_right,
        volume_force=volume_foce,
        num_collocation_points=num_collocation_points,
        num_points_per_bc=num_points_per_bc,
        bcs_overlap_distance=bcs_overlap_distance,
    )
    return create_training_dataset(config=config)


def assert_if_coordinates_are_inside_shape(coordinates: Tensor) -> bool:
    coordinates_list = coordinates.tolist()
    for one_coordinate in coordinates_list:
        if not shape.contains(shapely.Point(one_coordinate[0], one_coordinate[1])):
            return False
    return True


def generate_expected_x_parameters(num_points: int):
    return [
        (0, parameters[0].repeat(num_points, 1)),
        (1, parameters[1].repeat(num_points, 1)),
        (2, parameters[2].repeat(num_points, 1)),
        (3, parameters[3].repeat(num_points, 1)),
    ]


def test_len(sut: PlateWithHoleTrainingDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinates(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_collocation_points),
)
def test_sample_pde__x_parameters(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = volume_foce.repeat((num_collocation_points, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__x_coordinates(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_coor

    x_min_without_overlap = x_min + bcs_overlap_distance
    plate_length_without_overlap = plate_length - (2 * bcs_overlap_distance)

    expected = torch.tensor(
        [
            [x_max, y_min + 0 * plate_height],
            [x_max, y_min + 1 / 2 * plate_height],
            [x_max, y_min + 1 * plate_height],
            [x_min_without_overlap + 0 * plate_length_without_overlap, y_max],
            [x_min_without_overlap + 1 / 2 * plate_length_without_overlap, y_max],
            [x_min_without_overlap + plate_length_without_overlap, y_max],
            [x_min_without_overlap + 0 * plate_length_without_overlap, y_min],
            [x_min_without_overlap + 1 / 2 * plate_length_without_overlap, y_min],
            [x_min_without_overlap + plate_length_without_overlap, y_min],
            [
                origin_x
                - torch.cos(torch.deg2rad(torch.tensor(0 * 360))) * hole_radius,
                origin_y
                + torch.sin(torch.deg2rad(torch.tensor(0 * 360))) * hole_radius,
            ],
            [
                origin_x
                - torch.cos(torch.deg2rad(torch.tensor(1 / 3 * 360))) * hole_radius,
                origin_y
                + torch.sin(torch.deg2rad(torch.tensor(1 / 3 * 360))) * hole_radius,
            ],
            [
                origin_x
                - torch.cos(torch.deg2rad(torch.tensor(2 / 3 * 360))) * hole_radius,
                origin_y
                + torch.sin(torch.deg2rad(torch.tensor(2 / 3 * 360))) * hole_radius,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_points_traction_bcs),
)
def test_sample_traction_bc__x_youngs_modulus(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__normal(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.normal

    expected = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [
                torch.cos(torch.deg2rad(torch.tensor(0 * 360))),
                -torch.sin(torch.deg2rad(torch.tensor(0 * 360))),
            ],
            [
                torch.cos(torch.deg2rad(torch.tensor(1 / 3 * 360))),
                -torch.sin(torch.deg2rad(torch.tensor(1 / 3 * 360))),
            ],
            [
                torch.cos(torch.deg2rad(torch.tensor(2 / 3 * 360))),
                -torch.sin(torch.deg2rad(torch.tensor(2 / 3 * 360))),
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__area_fractions(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.area_frac

    hole_bc_length = 2.0 * math.pi * hole_radius
    expected = torch.tensor(
        [
            [plate_height / num_points_per_bc],
            [plate_height / num_points_per_bc],
            [plate_height / num_points_per_bc],
            [plate_length / num_points_per_bc],
            [plate_length / num_points_per_bc],
            [plate_length / num_points_per_bc],
            [plate_length / num_points_per_bc],
            [plate_length / num_points_per_bc],
            [plate_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__y_true(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.y_true

    expected = torch.stack(
        (
            traction_right,
            traction_right,
            traction_right,
            traction_top,
            traction_top,
            traction_top,
            traction_bottom,
            traction_bottom,
            traction_bottom,
            traction_hole,
            traction_hole,
            traction_hole,
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected)


### Test collate_func()
@pytest.fixture
def fake_batch() -> (
    list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ]
):
    sample_collocation_0 = TrainingData2DCollocation(
        x_coor=torch.tensor([[1.0, 1.0]]),
        x_params=torch.tensor([[1.1, 1.1]]),
        f=torch.tensor([[1.2, 1.2]]),
    )
    sample_traction_bc_0 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[3.0, 3.0]]),
        x_params=torch.tensor([[3.1, 3.1]]),
        normal=torch.tensor([[3.2, 3.2]]),
        area_frac=torch.tensor([[3.3]]),
        y_true=torch.tensor([[3.4, 3.4]]),
    )
    sample_collocation_1 = TrainingData2DCollocation(
        x_coor=torch.tensor([[10.0, 10.0]]),
        x_params=torch.tensor([[10.1, 10.1]]),
        f=torch.tensor([[10.2, 10.2]]),
    )
    sample_traction_bc_1 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[30.0, 30.0]]),
        x_params=torch.tensor([[30.1, 30.1]]),
        normal=torch.tensor([[30.2, 30.2]]),
        area_frac=torch.tensor([[30.3]]),
        y_true=torch.tensor([[30.4, 30.4]]),
    )
    return [
        (sample_collocation_0, sample_traction_bc_0),
        (sample_collocation_1, sample_traction_bc_1),
    ]


def test_batch_pde__x_coordinate(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.x_coor

    expected = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_parameters(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.x_params

    expected = torch.tensor([[1.1, 1.1], [10.1, 10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__volume_force(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.f

    expected = torch.tensor([[1.2, 1.2], [10.2, 10.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_coordinate(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.x_coor

    expected = torch.tensor([[3.0, 3.0], [30.0, 30.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_parameters(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.x_params

    expected = torch.tensor([[3.1, 3.1], [30.1, 30.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__normal(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.normal

    expected = torch.tensor([[3.2, 3.2], [30.2, 30.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__area_fraction(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.area_frac

    expected = torch.tensor([[3.3], [30.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__y_true(
    sut: PlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.y_true

    expected = torch.tensor([[3.4, 3.4], [30.4, 30.4]])
    torch.testing.assert_close(actual, expected)
