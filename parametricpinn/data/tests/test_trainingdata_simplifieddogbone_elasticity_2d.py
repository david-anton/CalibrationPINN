import math
from typing import TypeAlias

import pytest
import shapely
import torch

from parametricpinn.data.dataset import (
    TrainingData2DCollocation,
    TrainingData2DSymmetryBC,
    TrainingData2DTractionBC,
)
from parametricpinn.data.trainingdata_elasticity_2d import (
    SimplifiedDogBoneTrainingDataset2D,
    SimplifiedDogBoneTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.settings import set_default_dtype
from parametricpinn.types import ShapelyPolygon, Tensor

origin_x = 0
origin_y = 0
left_half_box_length = 60
right_half_box_length = 40
box_length = left_half_box_length + right_half_box_length
box_height = 30
half_box_height = box_height / 2
left_half_parallel_length = 45
right_half_parallel_length = 40
parallel_length = left_half_parallel_length + right_half_parallel_length
parallel_height = 20
half_parallel_height = parallel_height / 2
cut_parallel_height = half_box_height - half_parallel_height
tapered_radius = 25
plate_hole_radius = 4
angle_max_tapered = math.degrees(
    math.asin((box_length - parallel_length) / tapered_radius)
)
volume_foce = torch.tensor([10.0, 10.0], dtype=torch.float64)
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = 0.4
traction_right = torch.tensor([-100.0, 0.0], dtype=torch.float64)
traction_tapered = torch.tensor([0.0, 0.0], dtype=torch.float64)
traction_parralel = torch.tensor([0.0, 0.0], dtype=torch.float64)
traction_hole = torch.tensor([0.0, 0.0], dtype=torch.float64)
num_samples_per_parameter = 2
num_samples = num_samples_per_parameter**2
num_collocation_points = 32
num_points_per_bc = 3
num_traction_bcs = 6
num_symmetry_bcs = 2
num_points_traction_bcs = num_traction_bcs * num_points_per_bc
num_points_symmetry_bcs = num_symmetry_bcs * num_points_per_bc
bcs_overlap_angle_distance_left = 1.0
bcs_overlap_distance_parallel_right = 1.0

set_default_dtype(torch.float64)


### Test TrainingDataset
def _create_simplified_dog_bone_shape() -> ShapelyPolygon:
    box = shapely.box(
        -left_half_box_length,
        -half_box_height,
        right_half_box_length,
        half_box_height,
    )
    cut_parallel_top = shapely.box(
        -left_half_parallel_length,
        half_parallel_height,
        right_half_parallel_length,
        half_box_height,
    )
    cut_parallel_bottom = shapely.box(
        -left_half_parallel_length,
        -half_box_height,
        right_half_parallel_length,
        -half_parallel_height,
    )
    cut_tapered_top_left = shapely.Point(
        -left_half_parallel_length,
        half_parallel_height + tapered_radius,
    ).buffer(tapered_radius)
    cut_tapered_bottom_left = shapely.Point(
        -left_half_parallel_length,
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
        - cut_tapered_bottom_left
        - plate_hole
    )
    return dog_bone


shape = _create_simplified_dog_bone_shape()


@pytest.fixture
def sut() -> SimplifiedDogBoneTrainingDataset2D:
    config = SimplifiedDogBoneTrainingDataset2DConfig(
        traction_right=traction_right,
        volume_force=volume_foce,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        min_poissons_ratio=min_poissons_ratio,
        max_poissons_ratio=max_poissons_ratio,
        num_collocation_points=num_collocation_points,
        num_points_per_bc=num_points_per_bc,
        num_samples_per_parameter=num_samples_per_parameter,
        bcs_overlap_distance_parallel_right=bcs_overlap_distance_parallel_right,
        bcs_overlap_angle_distance_left=bcs_overlap_angle_distance_left,
    )
    return create_training_dataset(config=config)


def assert_if_coordinates_are_inside_shape(coordinates: Tensor) -> bool:
    coordinates_list = coordinates.tolist()
    for one_coordinate in coordinates_list:
        if not shape.contains(shapely.Point(one_coordinate[0], one_coordinate[1])):
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


def test_len(sut: SimplifiedDogBoneTrainingDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinates(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_collocation_points),
)
def test_sample_pde__x_youngs_modulus(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_collocation_points),
)
def test_sample_pde__x_poissons_ratio(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.x_nu

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    sample_pde, _, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = volume_foce.repeat((num_collocation_points, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__x_coordinates(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc, _ = sut[idx_sample]

    actual = sample_traction_bc.x_coor

    max_angle = angle_max_tapered - bcs_overlap_angle_distance_left
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
    corrected_parallel_length = parallel_length - bcs_overlap_distance_parallel_right
    expected = torch.tensor(
        [
            # right
            [right_half_box_length, -half_parallel_height],
            [right_half_box_length, 0.0],
            [right_half_box_length, half_parallel_height],
            # top
            [
                -left_half_parallel_length - abs_max_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            [
                -left_half_parallel_length - abs_half_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [-left_half_parallel_length, half_parallel_height],
            [
                -left_half_parallel_length + 1 / 3 * corrected_parallel_length,
                half_parallel_height,
            ],
            [
                -left_half_parallel_length + 2 / 3 * corrected_parallel_length,
                half_parallel_height,
            ],
            [
                -left_half_parallel_length + 3 / 3 * corrected_parallel_length,
                half_parallel_height,
            ],
            # bottom
            [
                -left_half_parallel_length - abs_max_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            [
                -left_half_parallel_length - abs_half_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [-left_half_parallel_length, -half_parallel_height],
            [
                -left_half_parallel_length + 1 / 3 * corrected_parallel_length,
                -half_parallel_height,
            ],
            [
                -left_half_parallel_length + 2 / 3 * corrected_parallel_length,
                -half_parallel_height,
            ],
            [
                -left_half_parallel_length + 3 / 3 * corrected_parallel_length,
                -half_parallel_height,
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
    generate_expected_x_youngs_modulus(num_points_traction_bcs),
)
def test_sample_traction_bc__x_youngs_modulus(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_traction_bc, _ = sut[idx_sample]

    actual = sample_traction_bc.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_points_traction_bcs),
)
def test_sample_traction_bc__x_poissons_ratio(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_traction_bc, _ = sut[idx_sample]

    actual = sample_traction_bc.x_nu
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__normal(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc, _ = sut[idx_sample]

    actual = sample_traction_bc.normal

    max_angle = angle_max_tapered - bcs_overlap_angle_distance_left
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
            # bottom
            [abs_normal_max_angle_tapered_x, -abs_normal_max_angle_tapered_y],
            [abs_normal_half_angle_tapered_x, -abs_normal_half_angle_tapered_y],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
            [0.0, -1.0],
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
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc, _ = sut[idx_sample]

    actual = sample_traction_bc.area_frac

    plate_hole_length = 2.0 * math.pi * plate_hole_radius
    tapered_length = (angle_max_tapered / 360) * 2.0 * math.pi * tapered_radius

    expected = torch.tensor(
        [
            # right
            [parallel_height / num_points_per_bc],
            [parallel_height / num_points_per_bc],
            [parallel_height / num_points_per_bc],
            # top
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            # bottom
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [tapered_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            [parallel_length / num_points_per_bc],
            # plate hole
            [plate_hole_length / num_points_per_bc],
            [plate_hole_length / num_points_per_bc],
            [plate_hole_length / num_points_per_bc],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_traction_bc__y_true(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, sample_traction_bc, _ = sut[idx_sample]

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
            # bottom
            traction_tapered,
            traction_tapered,
            traction_tapered,
            traction_parralel,
            traction_parralel,
            traction_parralel,
            # plate hole
            traction_hole,
            traction_hole,
            traction_hole,
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_symmetry_bc__x_coordinates_1(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_symmetry_bc = sut[idx_sample]

    actual = sample_symmetry_bc.x_coor_1

    max_angle = angle_max_tapered - bcs_overlap_angle_distance_left
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
    corrected_parallel_length = parallel_length - bcs_overlap_distance_parallel_right
    expected = torch.tensor(
        [
            # top
            [
                -left_half_parallel_length - abs_max_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            [
                -left_half_parallel_length - abs_half_angle_radial_component_tapered_x,
                half_parallel_height
                + (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [-left_half_parallel_length, half_parallel_height],
            [
                -left_half_parallel_length + 1 / 3 * corrected_parallel_length,
                half_parallel_height,
            ],
            [
                -left_half_parallel_length + 2 / 3 * corrected_parallel_length,
                half_parallel_height,
            ],
            [
                -left_half_parallel_length + 3 / 3 * corrected_parallel_length,
                half_parallel_height,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_symmetry_bc__x_coordinates_2(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int
) -> None:
    _, _, sample_symmetry_bc = sut[idx_sample]

    actual = sample_symmetry_bc.x_coor_2

    max_angle = angle_max_tapered - bcs_overlap_angle_distance_left
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
    corrected_parallel_length = parallel_length - bcs_overlap_distance_parallel_right
    expected = torch.tensor(
        [
            # bottom
            [
                -left_half_parallel_length - abs_max_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_max_angle_radial_component_tapered_y),
            ],
            [
                -left_half_parallel_length - abs_half_angle_radial_component_tapered_x,
                -half_parallel_height
                - (tapered_radius - abs_half_angle_radial_component_tapered_y),
            ],
            [-left_half_parallel_length, -half_parallel_height],
            [
                -left_half_parallel_length + 1 / 3 * corrected_parallel_length,
                -half_parallel_height,
            ],
            [
                -left_half_parallel_length + 2 / 3 * corrected_parallel_length,
                -half_parallel_height,
            ],
            [
                -left_half_parallel_length + 3 / 3 * corrected_parallel_length,
                -half_parallel_height,
            ],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_symmetry_bcs),
)
def test_sample_symmetry_bc__x_youngs_modulus(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, _, sample_symmetry_bc = sut[idx_sample]

    actual = sample_symmetry_bc.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_points_symmetry_bcs),
)
def test_sample_symmetry_bc__x_poissons_ratio(
    sut: SimplifiedDogBoneTrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, _, sample_symmetry_bc = sut[idx_sample]

    actual = sample_symmetry_bc.x_nu
    torch.testing.assert_close(actual, expected)


### Test collate_func()
FakeBatch: TypeAlias = list[
    tuple[
        TrainingData2DCollocation,
        TrainingData2DTractionBC,
        TrainingData2DSymmetryBC,
    ]
]


@pytest.fixture
def fake_batch() -> FakeBatch:
    sample_collocation_0 = TrainingData2DCollocation(
        x_coor=torch.tensor([[1.0, 1.0]]),
        x_E=torch.tensor([[1.1]]),
        x_nu=torch.tensor([[1.2]]),
        f=torch.tensor([[1.3]]),
    )
    sample_traction_bc_0 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[2.0, 2.0]]),
        x_E=torch.tensor([[2.1]]),
        x_nu=torch.tensor([[2.2]]),
        normal=torch.tensor([[2.3, 2.3]]),
        area_frac=torch.tensor([[2.4]]),
        y_true=torch.tensor([[2.5, 2.5]]),
    )
    sample_symmetry_bc_0 = TrainingData2DSymmetryBC(
        x_coor_1=torch.tensor([[3.0, 3.0]]),
        x_coor_2=torch.tensor([[3.1, 3.1]]),
        x_E=torch.tensor([[3.2]]),
        x_nu=torch.tensor([[3.3]]),
    )
    sample_collocation_1 = TrainingData2DCollocation(
        x_coor=torch.tensor([[10.0, 10.0]]),
        x_E=torch.tensor([[10.1]]),
        x_nu=torch.tensor([[10.2]]),
        f=torch.tensor([[10.3]]),
    )
    sample_traction_bc_1 = TrainingData2DTractionBC(
        x_coor=torch.tensor([[20.0, 20.0]]),
        x_E=torch.tensor([[20.1]]),
        x_nu=torch.tensor([[20.2]]),
        normal=torch.tensor([[20.3, 20.3]]),
        area_frac=torch.tensor([[20.4]]),
        y_true=torch.tensor([[20.5, 20.5]]),
    )
    sample_symmetry_bc_1 = TrainingData2DSymmetryBC(
        x_coor_1=torch.tensor([[30.0, 30.0]]),
        x_coor_2=torch.tensor([[30.1, 30.1]]),
        x_E=torch.tensor([[30.2]]),
        x_nu=torch.tensor([[30.3]]),
    )
    return [
        (sample_collocation_0, sample_traction_bc_0, sample_symmetry_bc_0),
        (sample_collocation_1, sample_traction_bc_1, sample_symmetry_bc_1),
    ]


def test_batch_pde__x_coordinate(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    batch_pde, _, _ = collate_func(fake_batch)
    actual = batch_pde.x_coor

    expected = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_youngs_modulus(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    batch_pde, _, _ = collate_func(fake_batch)
    actual = batch_pde.x_E

    expected = torch.tensor([[1.1], [10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_poissons_ratio(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    batch_pde, _, _ = collate_func(fake_batch)
    actual = batch_pde.x_nu

    expected = torch.tensor([[1.2], [10.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__volume_force(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    batch_pde, _, _ = collate_func(fake_batch)
    actual = batch_pde.f

    expected = torch.tensor([[1.3], [10.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_coordinate(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc, _ = collate_func(fake_batch)
    actual = batch_traction_bc.x_coor

    expected = torch.tensor([[2.0, 2.0], [20.0, 20.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_youngs_modulus(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc, _ = collate_func(fake_batch)
    actual = batch_traction_bc.x_E

    expected = torch.tensor([[2.1], [20.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__x_poissons_ratio(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc, _ = collate_func(fake_batch)
    actual = batch_traction_bc.x_nu

    expected = torch.tensor([[2.2], [20.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__normal(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc, _ = collate_func(fake_batch)
    actual = batch_traction_bc.normal

    expected = torch.tensor([[2.3, 2.3], [20.3, 20.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__area_fraction(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc, _ = collate_func(fake_batch)
    actual = batch_traction_bc.area_frac

    expected = torch.tensor([[2.4], [20.4]])
    torch.testing.assert_close(actual, expected)


def test_batch_traction_bc__y_true(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, batch_traction_bc, _ = collate_func(fake_batch)
    actual = batch_traction_bc.y_true

    expected = torch.tensor([[2.5, 2.5], [20.5, 20.5]])
    torch.testing.assert_close(actual, expected)


def test_batch_symmetry_bc__x_coordinate_1(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, _, batch_symmetry_bc = collate_func(fake_batch)
    actual = batch_symmetry_bc.x_coor_1

    expected = torch.tensor([[3.0, 3.0], [30.0, 30.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_symmetry_bc__x_coordinate_2(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, _, batch_symmetry_bc = collate_func(fake_batch)
    actual = batch_symmetry_bc.x_coor_2

    expected = torch.tensor([[3.1, 3.1], [30.1, 30.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_symmetry_bc__x_youngs_modulus(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, _, batch_symmetry_bc = collate_func(fake_batch)
    actual = batch_symmetry_bc.x_E

    expected = torch.tensor([[3.2], [30.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_symmetry_bc__x_poissons_ratio(
    sut: SimplifiedDogBoneTrainingDataset2D,
    fake_batch: FakeBatch,
):
    collate_func = sut.get_collate_func()

    _, _, batch_symmetry_bc = collate_func(fake_batch)
    actual = batch_symmetry_bc.x_nu

    expected = torch.tensor([[3.3], [30.3]])
    torch.testing.assert_close(actual, expected)
