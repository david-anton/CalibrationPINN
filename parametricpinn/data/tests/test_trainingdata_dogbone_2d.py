import math

import pytest
import shapely
import torch

from parametricpinn.data.dataset import (
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
)
from parametricpinn.data.trainingdata_2d import (
    DogBoneTrainingDataset2D,
    DogBoneTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.settings import set_default_dtype
from parametricpinn.types import ShapelyPolygon, Tensor

origin_x = 0
origin_y = 0
box_length = 120
box_height = 30
half_box_length = box_length / 2
half_box_height = box_height / 2
parallel_length = 90
parallel_height = 20
half_parallel_length = parallel_length / 2
half_parallel_height = parallel_height / 2
cut_parallel_height = (box_height - parallel_height) / 2
tapered_radius = 25
plate_hole_radius = 4
angle_max_tapered = math.degrees(
    math.asin((half_box_length - half_parallel_length) / tapered_radius)
)
parameters = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
volume_foce = torch.tensor([10.0, 10.0], dtype=torch.float64)
traction_right = torch.tensor([-100.0, 0.0], dtype=torch.float64)
traction_tapered = torch.tensor([0.0, 0.0], dtype=torch.float64)
traction_parralel = torch.tensor([0.0, 0.0], dtype=torch.float64)
traction_hole = torch.tensor([0.0, 0.0], dtype=torch.float64)
num_samples = len(parameters)
num_collocation_points = 32
num_points_per_bc = 3
num_traction_bcs = 8
num_points_traction_bcs = num_traction_bcs * num_points_per_bc
bcs_overlap_angle_distance = 1.0

set_default_dtype(torch.float64)


### Test TrainingDataset
def _create_dog_bone_shape() -> ShapelyPolygon:
    box = shapely.box(
        -half_box_length,
        -half_box_height,
        half_box_length,
        half_box_height,
    )
    cut_parallel_top = shapely.box(
        -half_parallel_length,
        half_parallel_height,
        half_parallel_length,
        half_box_height,
    )
    cut_parallel_bottom = shapely.box(
        -half_parallel_length,
        -half_box_height,
        half_parallel_length,
        -half_parallel_height,
    )
    cut_tapered_top_left = shapely.Point(
        -half_parallel_length,
        half_parallel_height + tapered_radius,
    ).buffer(tapered_radius)
    cut_tapered_top_right = shapely.Point(
        half_parallel_length,
        half_parallel_height + tapered_radius,
    ).buffer(tapered_radius)
    cut_tapered_bottom_left = shapely.Point(
        -half_parallel_length,
        -half_parallel_height - tapered_radius,
    ).buffer(tapered_radius)
    cut_tapered_bottom_right = shapely.Point(
        half_parallel_length,
        -half_parallel_height - tapered_radius,
    ).buffer(tapered_radius)
    plate_hole = shapely.Point(
        origin_x,
        origin_y,
    ).buffer(plate_hole_radius)
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


shape = _create_dog_bone_shape()


@pytest.fixture
def sut() -> DogBoneTrainingDataset2D:
    config = DogBoneTrainingDataset2DConfig(
        parameters_samples=parameters,
        traction_right=traction_right,
        volume_force=volume_foce,
        num_collocation_points=num_collocation_points,
        num_points_per_bc=num_points_per_bc,
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
        (0, parameters[0].repeat(num_points, 1)),
        (1, parameters[1].repeat(num_points, 1)),
        (2, parameters[2].repeat(num_points, 1)),
        (3, parameters[3].repeat(num_points, 1)),
    ]


def test_len(sut: DogBoneTrainingDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinates(
    sut: DogBoneTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_collocation_points),
)
def test_sample_pde__x_parameters(
    sut: DogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(
    sut: DogBoneTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = volume_foce.repeat((num_collocation_points, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__x_coordinates(
    sut: DogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_coor

    max_angle = angle_max_tapered - bcs_overlap_angle_distance
    half_angle = max_angle / 2
    abs_max_angle_radial_component_tapered_x = (
        math.sin(math.radians(max_angle)) * tapered_radius
    )
    abs_max_angle_radial_component_tapered_y = (
        math.cos(math.radians(max_angle)) * tapered_radius
    )
    abs_half_angle_radial_component_tapered_x = (
        math.sin(math.radians(half_angle)) * tapered_radius
    )
    abs_half_angle_radial_component_tapered_y = (
        math.cos(math.radians(half_angle)) * tapered_radius
    )
    expected = torch.tensor(
        [
            # right
            [half_box_length, -half_box_height],
            [half_box_length, 0.0],
            [half_box_length, half_box_height],
            # top
            [
                -half_parallel_length - abs_max_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            [
                -half_parallel_length - abs_half_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [-half_parallel_length, half_parallel_height],
            [
                -half_parallel_length + 1 / 4 * parallel_length,
                half_parallel_height,
            ],
            [
                -half_parallel_length + 2 / 4 * parallel_length,
                half_parallel_height,
            ],
            [
                -half_parallel_length + 3 / 4 * parallel_length,
                half_parallel_height,
            ],
            [half_parallel_length, half_parallel_height],
            [
                half_parallel_length + abs_half_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [
                half_parallel_length + abs_max_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            # bottom
            [
                -half_parallel_length - abs_max_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            [
                -half_parallel_length - abs_half_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [-half_parallel_length, -half_parallel_height],
            [
                -half_parallel_length + 1 / 4 * parallel_length,
                -half_parallel_height,
            ],
            [
                -half_parallel_length + 2 / 4 * parallel_length,
                -half_parallel_height,
            ],
            [
                -half_parallel_length + 3 / 4 * parallel_length,
                -half_parallel_height,
            ],
            [half_parallel_length, -half_parallel_height],
            [
                half_parallel_length + abs_half_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [
                half_parallel_length + abs_max_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            # plate hole
            [
                origin_x - math.cos(math.radians(0 / 3 * 360)) * plate_hole_radius,
                origin_y + math.sin(math.radians(0 / 3 * 360)) * plate_hole_radius,
            ],
            [
                origin_x - math.cos(math.radians(1 / 3 * 360)) * plate_hole_radius,
                origin_y + math.sin(math.radians(1 / 3 * 360)) * plate_hole_radius,
            ],
            [
                origin_x - math.cos(math.radians(2 / 3 * 360)) * plate_hole_radius,
                origin_y + math.sin(math.radians(2 / 3 * 360)) * plate_hole_radius,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_parameters(num_points_traction_bcs),
)
def test_sample_traction_bc__x_parameters(
    sut: DogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.x_params

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__normal(
    sut: DogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.normal

    max_angle = angle_max_tapered - bcs_overlap_angle_distance
    half_angle = max_angle / 2
    abs_normal_max_angle_tapered_x = math.sin(math.radians(max_angle))
    abs_normal_max_angle_tapered_y = math.cos(math.radians(max_angle))
    abs_normal_half_angle_tapered_x = math.sin(math.radians(half_angle))
    abs_normal_half_angle_tapered_y = math.cos(math.radians(half_angle))

    expected = torch.tensor(
        [
            # right
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            # top
            [abs_normal_max_angle_tapered_x, abs_normal_max_angle_tapered_y],
            [abs_normal_half_angle_tapered_x, abs_normal_half_angle_tapered_y],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [-abs_normal_half_angle_tapered_x, abs_normal_half_angle_tapered_y],
            [-abs_normal_max_angle_tapered_x, abs_normal_max_angle_tapered_y],
            # bottom
            [abs_normal_max_angle_tapered_x, -abs_normal_max_angle_tapered_y],
            [abs_normal_half_angle_tapered_x, -abs_normal_half_angle_tapered_y],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [-abs_normal_half_angle_tapered_x, -abs_normal_half_angle_tapered_y],
            [-abs_normal_max_angle_tapered_x, -abs_normal_max_angle_tapered_y],
            # plate hole
            [
                math.cos(math.radians(0 / 3 * 360)),
                -math.sin(math.radians(0 / 3 * 360)),
            ],
            [
                math.cos(math.radians(1 / 3 * 360)),
                -math.sin(math.radians(1 / 3 * 360)),
            ],
            [
                math.cos(math.radians(2 / 3 * 360)),
                -math.sin(math.radians(2 / 3 * 360)),
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__area_fractions(
    sut: DogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.area_frac

    plate_hole_length = 2.0 * math.pi * plate_hole_radius
    tapered_length = (angle_max_tapered / 360) * 2.0 * math.pi * tapered_radius

    expected = torch.tensor(
        [
            # right
            [box_height / num_points_per_bc],
            [box_height / num_points_per_bc],
            [box_height / num_points_per_bc],
            # top
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            # bottom
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            # plate hole
            [plate_hole_length / num_points_per_bc],
            [plate_hole_length / num_points_per_bc],
            [plate_hole_length / num_points_per_bc],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__y_true(
    sut: DogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc = sut[idx_sample]

    actual = sample_traction_bc.y_true

    expected = torch.stack(
        (
            # right
            traction_right,
            traction_right,
            traction_right,
            # top
            traction_tapered,
            traction_tapered,
            traction_tapered,
            traction_parralel,
            traction_parralel,
            traction_parralel,
            traction_tapered,
            traction_tapered,
            traction_tapered,
            # bottom
            traction_tapered,
            traction_tapered,
            traction_tapered,
            traction_parralel,
            traction_parralel,
            traction_parralel,
            traction_tapered,
            traction_tapered,
            traction_tapered,
            # plate hole
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
    sut: DogBoneTrainingDataset2D,
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


def test_batch_pde__x_params(
    sut: DogBoneTrainingDataset2D,
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
    sut: DogBoneTrainingDataset2D,
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
    sut: DogBoneTrainingDataset2D,
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


def test_batch_traction_bc__x_params(
    sut: DogBoneTrainingDataset2D,
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
    sut: DogBoneTrainingDataset2D,
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
    sut: DogBoneTrainingDataset2D,
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
    sut: DogBoneTrainingDataset2D,
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
