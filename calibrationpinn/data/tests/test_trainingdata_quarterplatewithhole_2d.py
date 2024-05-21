import math

import pytest
import shapely
import torch

from calibrationpinn.data.dataset import (
    TrainingData2DCollocation,
    TrainingData2DStressBC,
    TrainingData2DTractionBC,
)
from calibrationpinn.data.trainingdata_2d import (
    QuarterPlateWithHoleTrainingDataset2D,
    QuarterPlateWithHoleTrainingDataset2DConfig,
    create_training_dataset,
)
from calibrationpinn.settings import set_default_dtype
from calibrationpinn.types import ShapelyPolygon, Tensor

edge_length = 10.0
radius = 1.0
x_min = -edge_length
x_max = 0.0
y_min = 0.0
y_max = edge_length
parameters = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
volume_foce = torch.tensor([10.0, 10.0], dtype=torch.float64)
traction_left = torch.tensor([-100.0, 0.0], dtype=torch.float64)
traction_top = torch.tensor([0.0, 0.0], dtype=torch.float64)
traction_hole = torch.tensor([0.0, 0.0], dtype=torch.float64)
num_samples = len(parameters)
num_collocation_points = 32
num_points_per_bc = 3
num_stress_bcs = 2
num_points_stress_bcs = num_stress_bcs * num_points_per_bc
num_traction_bcs = 3
num_points_traction_bcs = num_traction_bcs * num_points_per_bc
bcs_overlap_distance = 1.0
bcs_overlap_angle_distance = 2.0

set_default_dtype(torch.float64)


### Test TrainingDataset
def _create_plate_with_hole_shape() -> ShapelyPolygon:
    plate = shapely.box(x_min, y_min, x_max, y_max)
    hole = shapely.Point(0, 0).buffer(radius)
    return shapely.difference(plate, hole)


shape = _create_plate_with_hole_shape()


@pytest.fixture
def sut() -> QuarterPlateWithHoleTrainingDataset2D:
    config = QuarterPlateWithHoleTrainingDataset2DConfig(
        parameters_samples=parameters,
        edge_length=edge_length,
        radius=radius,
        traction_left=traction_left,
        volume_force=volume_foce,
        num_collocation_points=num_collocation_points,
        num_points_per_bc=num_points_per_bc,
        bcs_overlap_distance=bcs_overlap_distance,
        bcs_overlap_angle_distance=bcs_overlap_angle_distance,
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
        (0, parameters[0].repeat((num_points, 1))),
        (1, parameters[1].repeat((num_points, 1))),
        (2, parameters[2].repeat((num_points, 1))),
        (3, parameters[3].repeat((num_points, 1))),
    ]


def test_len(sut: QuarterPlateWithHoleTrainingDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinates(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_collocation_points),
)
def test_sample_pde__x_parameters(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = volume_foce.repeat((num_collocation_points, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_stress_bc__x_coordinates(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_stress_bc, _ = sut[idx_sample]

    actual = sample_stress_bc.x_coor

    expected = torch.tensor(
        [
            [0.0, radius],
            [0.0, radius + 1 / 2 * (edge_length - radius)],
            [0.0, edge_length],
            [-edge_length, 0.0],
            [-radius - 1 / 2 * (edge_length - radius), 0.0],
            [-radius, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_points_stress_bcs),
)
def test_sample_stress_bc__x_parameters(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_stress_bc, _ = sut[idx_sample]

    actual = sample_stress_bc.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__x_coordinates(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_coor

    y_min = bcs_overlap_distance
    x_min = -edge_length + bcs_overlap_distance
    x_max = -bcs_overlap_distance
    min_angle = bcs_overlap_angle_distance
    max_angle = 90 - bcs_overlap_angle_distance
    half_angle = 90 / 2

    expected = torch.tensor(
        [
            [-edge_length, y_min],
            [-edge_length, y_min + (1 / 2 * (edge_length - y_min))],
            [-edge_length, edge_length],
            [x_min, edge_length],
            [-1 / 2 * edge_length, edge_length],
            [x_max, edge_length],
            [
                -torch.cos(torch.deg2rad(torch.tensor(min_angle))) * radius,
                torch.sin(torch.deg2rad(torch.tensor(min_angle))) * radius,
            ],
            [
                -torch.cos(torch.deg2rad(torch.tensor(half_angle))) * radius,
                torch.sin(torch.deg2rad(torch.tensor(half_angle))) * radius,
            ],
            [
                -torch.cos(torch.deg2rad(torch.tensor(max_angle))) * radius,
                torch.sin(torch.deg2rad(torch.tensor(max_angle))) * radius,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_points_traction_bcs),
)
def test_sample_traction_bc__x_parameters(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__normal(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.normal

    min_angle = bcs_overlap_angle_distance
    max_angle = 90 - bcs_overlap_angle_distance
    half_angle = 90 / 2

    expected = torch.tensor(
        [
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [
                torch.cos(torch.deg2rad(torch.tensor(min_angle))),
                -torch.sin(torch.deg2rad(torch.tensor(min_angle))),
            ],
            [
                torch.cos(torch.deg2rad(torch.tensor(half_angle))),
                -torch.sin(torch.deg2rad(torch.tensor(half_angle))),
            ],
            [
                torch.cos(torch.deg2rad(torch.tensor(max_angle))),
                -torch.sin(torch.deg2rad(torch.tensor(max_angle))),
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__area_fractions(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.area_frac

    hole_bc_length = 1 / 4 * 2.0 * math.pi * radius
    expected = torch.tensor(
        [
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [edge_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
            [hole_bc_length / num_points_per_bc],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__y_true(
    sut: QuarterPlateWithHoleTrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.y_true

    expected = torch.stack(
        (
            traction_left,
            traction_left,
            traction_left,
            traction_top,
            traction_top,
            traction_top,
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
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ]
):
    sample_collocation_0 = TrainingData2DCollocation(
        x_coor=torch.tensor([[1.0, 1.0]]),
        x_params=torch.tensor([[1.1, 1.1]]),
        f=torch.tensor([[1.2, 1.2]]),
    )
    sample_stress_bc_0 = TrainingData2DStressBC(
        x_coor=torch.tensor([[2.0, 2.0]]),
        x_params=torch.tensor([[2.1, 2.1]]),
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
    sample_stress_bc_1 = TrainingData2DStressBC(
        x_coor=torch.tensor([[20.0, 20.0]]),
        x_params=torch.tensor([[20.1, 20.1]]),
    )
    sample_traction_bc_1 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[30.0, 30.0]]),
        x_params=torch.tensor([[30.1, 30.1]]),
        normal=torch.tensor([[30.2, 30.2]]),
        area_frac=torch.tensor([[30.3]]),
        y_true=torch.tensor([[30.4, 30.4]]),
    )
    return [
        (sample_collocation_0, sample_stress_bc_0, sample_traction_bc_0),
        (sample_collocation_1, sample_stress_bc_1, sample_traction_bc_1),
    ]


def test_batch_pde__x_coordinate(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    batch_pde, _, _ = collate_func(fake_batch)
    actual = batch_pde.x_coor

    expected = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_parameters(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    batch_pde, _, _ = collate_func(fake_batch)
    actual = batch_pde.x_params

    expected = torch.tensor([[1.1, 1.1], [10.1, 10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__volume_force(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    batch_pde, _, _ = collate_func(fake_batch)
    actual = batch_pde.f

    expected = torch.tensor([[1.2, 1.2], [10.2, 10.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__x_coordinate(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, batch_stress_bc, _ = collate_func(fake_batch)
    actual = batch_stress_bc.x_coor

    expected = torch.tensor([[2.0, 2.0], [20.0, 20.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__x_parameters(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, batch_stress_bc, _ = collate_func(fake_batch)
    actual = batch_stress_bc.x_params

    expected = torch.tensor([[2.1, 2.1], [20.1, 20.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_coordinate(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.x_coor

    expected = torch.tensor([[3.0, 3.0], [30.0, 30.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_parameters(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.x_params

    expected = torch.tensor([[3.1, 3.1], [30.1, 30.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__normal(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.normal

    expected = torch.tensor([[3.2, 3.2], [30.2, 30.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__area_fraction(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.area_frac

    expected = torch.tensor([[3.3], [30.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__y_true(
    sut: QuarterPlateWithHoleTrainingDataset2D,
    fake_batch: list[
        tuple[
            TrainingData2DCollocation,
            TrainingData2DStressBC,
            TrainingData2DTractionBC,
        ]
    ],
):
    collate_func = sut.get_collate_func()

    _, _, batch_traction_bc = collate_func(fake_batch)
    actual = batch_traction_bc.y_true

    expected = torch.tensor([[3.4, 3.4], [30.4, 30.4]])
    torch.testing.assert_close(actual, expected)
