import math

import pytest
import torch
from shapely.geometry import Point, Polygon, box

from parametricpinn.data.dataset import (
    TrainingData2DCollocation,
    TrainingData2DSymmetryBC,
    TrainingData2DTractionBC,
)
from parametricpinn.data.trainingdata_elasticity_2d import (
    PlateWithHoleTrainingDataset2D,
    PlateWithHoleTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.types import Tensor

plate_length = 20.0
plate_height = 10.0
hole_radius = 2.0
x_min = 0.0
x_max = plate_length
y_min = 0.0
y_max = plate_height
volume_foce = torch.tensor([10.0, 10.0])
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = 0.4
traction_right = torch.tensor([-100.0, 0.0])
traction_top = torch.tensor([0.0, 0.0])
traction_bottom = torch.tensor([0.0, 0.0])
traction_hole = torch.tensor([0.0, 0.0])
num_samples_per_parameter = 2
num_samples = num_samples_per_parameter**2
num_collocation_points = 3
num_points_per_bc = 3
num_traction_bcs = 4
num_points_traction_bcs = num_traction_bcs * num_points_per_bc


### Test TrainingDataset
def _create_plate_with_hole_shape() -> Polygon:
    plate = box(x_min, y_min, x_max, y_max)
    hole = Point(plate_length / 2, plate_height / 2).buffer(hole_radius)
    return plate.difference(hole)


shape = _create_plate_with_hole_shape()


@pytest.fixture
def sut() -> PlateWithHoleTrainingDataset2D:
    config = PlateWithHoleTrainingDataset2DConfig(
        plate_length=plate_length,
        plate_height=plate_height,
        hole_radius=hole_radius,
        traction_right=traction_right,
        volume_force=volume_foce,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        min_poissons_ratio=min_poissons_ratio,
        max_poissons_ratio=max_poissons_ratio,
        num_collocation_points=num_collocation_points,
        num_points_per_bc=num_points_per_bc,
        num_samples_per_parameter=num_samples_per_parameter,
    )
    return create_training_dataset(config=config)


def assert_if_coordinates_are_inside_shape(coordinates: Tensor) -> bool:
    coordinates_list = coordinates.tolist()
    for one_coordinate in coordinates_list:
        if not shape.contains(Point(one_coordinate[0], one_coordinate[1])):
            return False
    return True


def generate_expected_x_youngs_modulus(num_points: int):
    return [
        (0, torch.full((num_points, 1), min_youngs_modulus)),
        (1, torch.full((num_points, 1), min_youngs_modulus)),
        (2, torch.full((num_points, 1), max_youngs_modulus)),
        (3, torch.full((num_points, 1), max_youngs_modulus)),
    ]


def generate_expected_x_poissons_ratio(num_points: int):
    return [
        (0, torch.full((num_points, 1), min_poissons_ratio)),
        (1, torch.full((num_points, 1), max_poissons_ratio)),
        (2, torch.full((num_points, 1), min_poissons_ratio)),
        (3, torch.full((num_points, 1), max_poissons_ratio)),
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
    generate_expected_x_youngs_modulus(num_collocation_points),
)
def test_sample_pde__x_youngs_modulus(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_collocation_points),
)
def test_sample_pde__x_poissons_ratio(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_nu

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

    expected = torch.tensor(
        [
            [x_max, 0 * plate_height],
            [x_max, 1 / 2 * plate_height],
            [x_max, 1 * plate_height],
            [1 / 4 * plate_length, y_max],
            [2 / 4 * plate_length, y_max],
            [3 / 4 * plate_length, y_max],
            [1 / 4 * plate_length, y_min],
            [2 / 4 * plate_length, y_min],
            [3 / 4 * plate_length, y_min],
            [
                (plate_length / 2)
                - torch.cos(torch.deg2rad(torch.tensor(0 * 360))) * hole_radius,
                (plate_height / 2)
                + torch.sin(torch.deg2rad(torch.tensor(0 * 360))) * hole_radius,
            ],
            [
                (plate_length / 2)
                - torch.cos(torch.deg2rad(torch.tensor(1 / 3 * 360))) * hole_radius,
                (plate_height / 2)
                + torch.sin(torch.deg2rad(torch.tensor(1 / 3 * 360))) * hole_radius,
            ],
            [
                (plate_length / 2)
                - torch.cos(torch.deg2rad(torch.tensor(2 / 3 * 360))) * hole_radius,
                (plate_height / 2)
                + torch.sin(torch.deg2rad(torch.tensor(2 / 3 * 360))) * hole_radius,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_traction_bcs),
)
def test_sample_traction_bc__x_youngs_modulus(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_points_traction_bcs),
)
def test_sample_traction_bc__x_poissons_ratio(
    sut: PlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_nu
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
        x_E=torch.tensor([[1.1]]),
        x_nu=torch.tensor([[1.2]]),
        f=torch.tensor([[1.3]]),
    )
    sample_traction_bc_0 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[3.0, 3.0]]),
        x_E=torch.tensor([[3.1]]),
        x_nu=torch.tensor([[3.2]]),
        normal=torch.tensor([[3.3, 3.3]]),
        area_frac=torch.tensor([[3.4]]),
        y_true=torch.tensor([[3.5, 3.5]]),
    )
    sample_collocation_1 = TrainingData2DCollocation(
        x_coor=torch.tensor([[10.0, 10.0]]),
        x_E=torch.tensor([[10.1]]),
        x_nu=torch.tensor([[10.2]]),
        f=torch.tensor([[10.3]]),
    )
    sample_traction_bc_1 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[30.0, 30.0]]),
        x_E=torch.tensor([[30.1]]),
        x_nu=torch.tensor([[30.2]]),
        normal=torch.tensor([[30.3, 30.3]]),
        area_frac=torch.tensor([[30.4]]),
        y_true=torch.tensor([[30.5, 30.5]]),
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


def test_batch_pde__x_youngs_modulus(
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
    actual = batch_pde.x_E

    expected = torch.tensor([[1.1], [10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_poissons_ratio(
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
    actual = batch_pde.x_nu

    expected = torch.tensor([[1.2], [10.2]])
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

    expected = torch.tensor([[1.3], [10.3]])
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


def test_batch_traction_bc__x_youngs_modulus(
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
    actual = batch_traction_bc.x_E

    expected = torch.tensor([[3.1], [30.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_poissons_ratio(
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
    actual = batch_traction_bc.x_nu

    expected = torch.tensor([[3.2], [30.2]])
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

    expected = torch.tensor([[3.3, 3.3], [30.3, 30.3]])
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

    expected = torch.tensor([[3.4], [30.4]])
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

    expected = torch.tensor([[3.5, 3.5], [30.5, 30.5]])
    torch.testing.assert_close(actual, expected)
