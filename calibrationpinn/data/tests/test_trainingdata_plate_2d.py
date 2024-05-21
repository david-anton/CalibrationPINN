import math
from typing import TypeAlias

import pytest
import shapely
import torch

from calibrationpinn.data.dataset import (
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
)
from calibrationpinn.data.trainingdata_2d import (
    PlateTrainingDataset2D,
    PlateTrainingDataset2DConfig,
    create_training_dataset,
)
from calibrationpinn.settings import set_default_dtype
from calibrationpinn.types import ShapelyPolygon, Tensor

plate_length = 20.0
plate_height = 10.0
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
num_samples = len(parameters)
num_collocation_points = 32
num_points_per_bc = 3
num_traction_bcs = 3
num_points_traction_bcs = num_traction_bcs * num_points_per_bc
bcs_overlap_distance = 1.0

set_default_dtype(torch.float64)


### Test TrainingDataset
def _create_plate_shape() -> ShapelyPolygon:
    plate = shapely.box(x_min, y_min, x_max, y_max)
    return plate


shape = _create_plate_shape()


@pytest.fixture
def sut() -> PlateTrainingDataset2D:
    config = PlateTrainingDataset2DConfig(
        parameters_samples=parameters,
        plate_length=plate_length,
        plate_height=plate_height,
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


def test_len(sut: PlateTrainingDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinates(
    sut: PlateTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_collocation_points),
)
def test_sample_pde__x_parameters(
    sut: PlateTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(sut: PlateTrainingDataset2D, idx_sample: int) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = volume_foce.repeat((num_collocation_points, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__x_coordinates(
    sut: PlateTrainingDataset2D, idx_sample: int
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
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_points_traction_bcs),
)
def test_sample_traction_bc__x_parameters(
    sut: PlateTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__normal(
    sut: PlateTrainingDataset2D, idx_sample: int
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
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__area_fractions(
    sut: PlateTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.area_frac

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
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__y_true(
    sut: PlateTrainingDataset2D, idx_sample: int
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
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected)


### Test collate_func()
FakeBatch: TypeAlias = list[
    tuple[
        TrainingData2DCollocation,
        TrainingData2DTractionBC,
    ]
]


@pytest.fixture
def fake_batch() -> FakeBatch:
    sample_collocation_0 = TrainingData2DCollocation(
        x_coor=torch.tensor([[1.0, 1.0]]),
        x_params=torch.tensor([[1.1, 1.1]]),
        f=torch.tensor([[1.2, 1.2]]),
    )
    sample_traction_bc_0 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[2.0, 2.0]]),
        x_params=torch.tensor([[2.1, 2.1]]),
        normal=torch.tensor([[2.2, 2.2]]),
        area_frac=torch.tensor([[2.3]]),
        y_true=torch.tensor([[2.4, 2.4]]),
    )
    sample_collocation_1 = TrainingData2DCollocation(
        x_coor=torch.tensor([[10.0, 10.0]]),
        x_params=torch.tensor([[10.1, 10.1]]),
        f=torch.tensor([[10.2, 10.2]]),
    )
    sample_traction_bc_1 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[20.0, 20.0]]),
        x_params=torch.tensor([[20.1, 20.1]]),
        normal=torch.tensor([[20.2, 20.2]]),
        area_frac=torch.tensor([[20.3]]),
        y_true=torch.tensor([[20.4, 20.4]]),
    )
    return [
        (sample_collocation_0, sample_traction_bc_0),
        (sample_collocation_1, sample_traction_bc_1),
    ]


def test_batch_pde__x_coordinate(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.x_coor

    expected = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_parameters(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.x_params

    expected = torch.tensor([[1.1, 1.1], [10.1, 10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__volume_force(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.f

    expected = torch.tensor([[1.2, 1.2], [10.2, 10.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_coordinate(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.x_coor

    expected = torch.tensor([[2.0, 2.0], [20.0, 20.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_parameters(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.x_params

    expected = torch.tensor([[2.1, 2.1], [20.1, 20.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__normal(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.normal

    expected = torch.tensor([[2.2, 2.2], [20.2, 20.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__area_fraction(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.area_frac

    expected = torch.tensor([[2.3], [20.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__y_true(
    sut: PlateTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.y_true

    expected = torch.tensor([[2.4, 2.4], [20.4, 20.4]])
    torch.testing.assert_close(actual, expected)
