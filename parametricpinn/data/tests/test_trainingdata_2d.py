import pytest
import torch

from parametricpinn.data import (
    TrainingData2DPDE,
    TrainingData2DStressBC,
    TrainingDataset2D,
    collate_training_data_2D,
)
from parametricpinn.types import Tensor

traction_left = torch.tensor([1.0, 1.0])
normal_left = torch.tensor([2.0, 2.0])
volume_foce = torch.tensor([3.0, 3.0])
min_youngs_modulus = 5.0
max_youngs_modulus = 6.0
min_poissons_ratio = 7.0
max_poissons_ratio = 8.0
num_points_pde = 4
num_points_stress_bc = 3
num_samples_per_parameter = 2
num_samples = num_samples_per_parameter**2


class FakeGeometry2D:
    def create_random_points(self, num_points: int) -> Tensor:
        return torch.tensor([[0.0, 0.0], [10.0, 10.0]])

    def create_uniform_points_on_left_boundary(self, num_points):
        return torch.tensor([[0.0, 0.0], [0.0, 10.0]])


def generate_expected_x_youngs_modulus(num_points: int):
    return [
        (0, torch.tensor([min_youngs_modulus]).repeat(num_points, 1)),
        (1, torch.tensor([min_youngs_modulus]).repeat(num_points, 1)),
        (2, torch.tensor([max_youngs_modulus]).repeat(num_points, 1)),
        (3, torch.tensor([max_youngs_modulus]).repeat(num_points, 1)),
    ]


def generate_expected_x_poissons_ratio(num_points: int):
    return [
        (0, torch.tensor([min_poissons_ratio]).repeat(num_points, 1)),
        (1, torch.tensor([max_poissons_ratio]).repeat(num_points, 1)),
        (2, torch.tensor([min_poissons_ratio]).repeat(num_points, 1)),
        (3, torch.tensor([max_poissons_ratio]).repeat(num_points, 1)),
    ]


### Test TrainingDataset1D
@pytest.fixture
def sut() -> TrainingDataset2D:
    fake_geometry = FakeGeometry2D()
    return TrainingDataset2D(
        geometry=fake_geometry,
        traction_left=traction_left,
        normal_left=normal_left,
        volume_force=volume_foce,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        min_poissons_ratio=min_poissons_ratio,
        max_poissons_ratio=max_poissons_ratio,
        num_points_pde=num_points_pde,
        num_points_stress_bc=num_points_stress_bc,
        num_samples_per_parameter=num_samples_per_parameter,
    )


def test_len(sut: TrainingDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__x_coordinates(sut: TrainingDataset2D, idx_sample: int) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_coor

    expected = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_pde),
)
def test_sample_pde__x_youngs_modulus(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_E

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_points_pde),
)
def test_sample_pde__x_poissons_ratio(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.x_nu

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__volume_force(sut: TrainingDataset2D, idx_sample: int) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.f

    expected = volume_foce.repeat((num_points_pde, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_pde__y_true(sut: TrainingDataset2D, idx_sample: int) -> None:
    sample_pde, _ = sut[idx_sample]

    actual = sample_pde.y_true

    expected = torch.zeros((num_points_pde, 2))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_stress_bc__x_coordinates(
    sut: TrainingDataset2D, idx_sample: int
) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.x_coor

    expected = torch.tensor([[0.0, 0.0], [0.0, 10.0]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_youngs_modulus(num_points_stress_bc),
)
def test_sample_stress_bc__x_youngs_modulus(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.x_E
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("idx_sample", "expected"),
    generate_expected_x_poissons_ratio(num_points_stress_bc),
)
def test_sample_stress_bc__x_poissons_ratio(
    sut: TrainingDataset2D, idx_sample: int, expected: Tensor
) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.x_nu
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_stress_bc__normal(sut: TrainingDataset2D, idx_sample: int) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.normal

    expected = normal_left.repeat((num_points_stress_bc, 1))
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_sample_stress_bc__y_true(sut: TrainingDataset2D, idx_sample: int) -> None:
    _, sample_stress_bc = sut[idx_sample]

    actual = sample_stress_bc.y_true

    expected = traction_left.repeat((num_points_stress_bc, 1))
    torch.testing.assert_close(actual, expected)


### Test collate_training_data_2D()
@pytest.fixture
def fake_batch() -> list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]:
    sample_pde_0 = TrainingData2DPDE(
        x_coor=torch.tensor([[1.0, 1.0]]),
        x_E=torch.tensor([[1.1]]),
        x_nu=torch.tensor([[1.2]]),
        f=torch.tensor([[1.3]]),
        y_true=torch.tensor([[1.4]]),
    )
    sample_stress_bc_0 = TrainingData2DStressBC(
        x_coor=torch.tensor([[2.0, 2.0]]),
        x_E=torch.tensor([[2.1]]),
        x_nu=torch.tensor([[2.2]]),
        normal=torch.tensor([[2.3, 2.3]]),
        y_true=torch.tensor([[2.4, 2.4]]),
    )
    sample_pde_1 = TrainingData2DPDE(
        x_coor=torch.tensor([[10.0, 10.0]]),
        x_E=torch.tensor([[10.1]]),
        x_nu=torch.tensor([[10.2]]),
        f=torch.tensor([[10.3]]),
        y_true=torch.tensor([[10.4]]),
    )
    sample_stress_bc_1 = TrainingData2DStressBC(
        x_coor=torch.tensor([[20.0, 20.0]]),
        x_E=torch.tensor([[20.1]]),
        x_nu=torch.tensor([[20.2]]),
        normal=torch.tensor([[20.3, 20.3]]),
        y_true=torch.tensor([[20.4, 20.4]]),
    )
    return [(sample_pde_0, sample_stress_bc_0), (sample_pde_1, sample_stress_bc_1)]


def test_batch_pde__x_coordinate(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    batch_pde, _ = sut(fake_batch)
    actual = batch_pde.x_coor

    expected = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_youngs_modulus(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    batch_pde, _ = sut(fake_batch)
    actual = batch_pde.x_E

    expected = torch.tensor([[1.1], [10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__x_poissons_ratio(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    batch_pde, _ = sut(fake_batch)
    actual = batch_pde.x_nu

    expected = torch.tensor([[1.2], [10.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__volume_force(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    batch_pde, _ = sut(fake_batch)
    actual = batch_pde.f

    expected = torch.tensor([[1.3], [10.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__y_true(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    batch_pde, _ = sut(fake_batch)
    actual = batch_pde.y_true

    expected = torch.tensor([[1.4], [10.4]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__x_coordinate(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    _, batch_stress_bc = sut(fake_batch)
    actual = batch_stress_bc.x_coor

    expected = torch.tensor([[2.0, 2.0], [20.0, 20.0]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__x_youngs_modulus(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    _, batch_stress_bc = sut(fake_batch)
    actual = batch_stress_bc.x_E

    expected = torch.tensor([[2.1], [20.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__x_poissons_ratio(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    _, batch_stress_bc = sut(fake_batch)
    actual = batch_stress_bc.x_nu

    expected = torch.tensor([[2.2], [20.2]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__normal(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    _, batch_stress_bc = sut(fake_batch)
    actual = batch_stress_bc.normal

    expected = torch.tensor([[2.3, 2.3], [20.3, 20.3]])
    torch.testing.assert_close(actual, expected)


def test_batch_stress_bc__y_true(
    fake_batch: list[tuple[TrainingData2DPDE, TrainingData2DStressBC]]
):
    sut = collate_training_data_2D

    _, batch_stress_bc = sut(fake_batch)
    actual = batch_stress_bc.y_true

    expected = torch.tensor([[2.4, 2.4], [20.4, 20.4]])
    torch.testing.assert_close(actual, expected)
