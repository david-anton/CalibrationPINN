import pytest
import torch

from parametricpinn.data.dataset import (
    TrainingData1DCollocation,
    TrainingData1DTractionBC,
)
from parametricpinn.data.trainingdata_linearelasticity_1d import (
    StretchedRodTrainingDataset1D,
    StretchedRodTrainingDataset1DConfig,
    create_training_dataset,
)
from parametricpinn.types import Tensor

length = 10.0
traction = 1.0
volume_force = 2.0
min_youngs_modulus = 3.0
max_youngs_modulus = 4.0
num_points_pde = 2
num_points_stress_bc = 1
num_samples = 3


def generate_expected_x_youngs_modulus(num_points: int):
    return [
        (0, torch.full((num_points, 1), min_youngs_modulus)),
        (
            1,
            torch.full(
                (num_points, 1),
                (min_youngs_modulus + max_youngs_modulus) / 2,
            ),
        ),
        (2, torch.full((num_points, 1), max_youngs_modulus)),
    ]


### Test TrainingDataset
@pytest.fixture
def sut() -> StretchedRodTrainingDataset1D:
    config = StretchedRodTrainingDataset1DConfig(
        length=length,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples,
    )
    return create_training_dataset(config=config)


def assert_if_coordinates_are_inside_shape(coordinates: Tensor) -> bool:
    coordinates_list = coordinates.tolist()
    for one_coordinate in coordinates_list:
        if not (one_coordinate[0] >= 0.0 and one_coordinate[0] <= length):
            return False
    return True


def test_len(sut: StretchedRodTrainingDataset1D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinate(
    sut: StretchedRodTrainingDataset1D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_pde),
)
def test_sample_pde__x_youngs_modulus(
    sut: StretchedRodTrainingDataset1D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(
    sut: StretchedRodTrainingDataset1D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = torch.full((num_points_pde, 1), volume_force)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__y_true(
    sut: StretchedRodTrainingDataset1D, idx_sample: int
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.y_true

    expected = torch.zeros((num_points_pde, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_stress_bc__x_coordinate(
    sut: StretchedRodTrainingDataset1D, idx_sample: int
) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.x_coor

    assert assert_if_coordinates_are_inside_shape(actual)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_stress_bc),
)
def test_sample_stress_bc__x_youngs_modulus(
    sut: StretchedRodTrainingDataset1D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_stress_bc__y_true(
    sut: StretchedRodTrainingDataset1D, idx_sample: int
) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.y_true

    expected = torch.full((num_points_stress_bc, 1), traction)
    torch.testing.assert_close(actual, expected)


### Test collate_func()
@pytest.fixture
def fake_batch() -> list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]]:
    sample_pde_0 = TrainingData1DCollocation(
        x_coor=torch.tensor([[1.0]]),
        x_E=torch.tensor([[1.1]]),
        f=torch.tensor([[1.2]]),
        y_true=torch.tensor([[1.3]]),
    )
    sample_stress_bc_0 = TrainingData1DTractionBC(
        x_coor=torch.tensor([[2.0]]),
        x_E=torch.tensor([[2.1]]),
        y_true=torch.tensor([[2.2]]),
    )
    sample_pde_1 = TrainingData1DCollocation(
        x_coor=torch.tensor([[10.0]]),
        x_E=torch.tensor([[10.1]]),
        f=torch.tensor([[10.2]]),
        y_true=torch.tensor([[10.3]]),
    )
    sample_stress_bc_1 = TrainingData1DTractionBC(
        x_coor=torch.tensor([[20.0]]),
        x_E=torch.tensor([[20.1]]),
        y_true=torch.tensor([[20.2]]),
    )
    return [(sample_pde_0, sample_stress_bc_0), (sample_pde_1, sample_stress_bc_1)]


def test_batch_pde__x_coordinate(
    sut: StretchedRodTrainingDataset1D,
    fake_batch: list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]],
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.x_coor

    expected = torch.tensor([[1.0], [10.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_youngs_modulus(
    sut: StretchedRodTrainingDataset1D,
    fake_batch: list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]],
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.x_E

    expected = torch.tensor([[1.1], [10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__volume_force(
    sut: StretchedRodTrainingDataset1D,
    fake_batch: list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]],
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.f

    expected = torch.tensor([[1.2], [10.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__y_true(
    sut: StretchedRodTrainingDataset1D,
    fake_batch: list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]],
):
    collate_func = sut.get_collate_func()

    batch_pde, _ = collate_func(fake_batch)
    actual = batch_pde.y_true

    expected = torch.tensor([[1.3], [10.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__x_coordinate(
    sut: StretchedRodTrainingDataset1D,
    fake_batch: list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]],
):
    collate_func = sut.get_collate_func()

    _, batch_stress_bc = collate_func(fake_batch)
    actual = batch_stress_bc.x_coor

    expected = torch.tensor([[2.0], [20.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__x_youngs_modulus(
    sut: StretchedRodTrainingDataset1D,
    fake_batch: list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]],
):
    collate_func = sut.get_collate_func()

    _, batch_stress_bc = collate_func(fake_batch)
    actual = batch_stress_bc.x_E

    expected = torch.tensor([[2.1], [20.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__y_true(
    sut: StretchedRodTrainingDataset1D,
    fake_batch: list[tuple[TrainingData1DCollocation, TrainingData1DTractionBC]],
):
    collate_func = sut.get_collate_func()

    _, batch_stress_bc = collate_func(fake_batch)
    actual = batch_stress_bc.y_true

    expected = torch.tensor([[2.2], [20.2]])
    torch.testing.assert_close(actual, expected)
