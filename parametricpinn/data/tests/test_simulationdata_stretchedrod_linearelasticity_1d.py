import pytest
import torch

from parametricpinn.data.simulationdata_linearelasticity_1d import (
    StretchedRodSimulationDatasetLinearElasticity1D,
    StretchedRodSimulationDatasetLinearElasticity1DConfig,
    calculate_linear_elastic_displacements_solution,
    create_simulation_dataset,
)
from parametricpinn.settings import set_seed
from parametricpinn.types import Tensor

length = 10.0
traction = 1.0
volume_force = 2.0
min_youngs_modulus = 3.0
max_youngs_modulus = 4.0
num_points = 3
num_samples = 3
random_seed = 0


### Test calculate_displacement_solution()
@pytest.mark.parametrize(
    ("coordinate", "expected"),
    [
        (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[1.0]]), torch.tensor([[10.0]])),
        (torch.tensor([[2.0]]), torch.tensor([[18.0]])),
    ],
)
def test_calculate_displacements_solution(coordinate: Tensor, expected: Tensor) -> None:
    sut = calculate_linear_elastic_displacements_solution
    length = 4.0
    youngs_modulus = 1.0
    traction = 3.0
    volume_force = 2.0

    actual = sut(
        coordinates=coordinate,
        length=length,
        youngs_modulus=youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    torch.testing.assert_close(actual, expected)


### Test ValidationDataset
@pytest.fixture
def sut() -> StretchedRodSimulationDatasetLinearElasticity1D:
    set_seed(random_seed)
    config = StretchedRodSimulationDatasetLinearElasticity1DConfig(
        length=length,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points=num_points,
        num_samples=num_samples,
    )
    return create_simulation_dataset(config=config)


@pytest.fixture
def expected_input_sample() -> tuple[list[Tensor], list[Tensor]]:
    # The random numbers must be generated in the same order as in the system under test.
    set_seed(random_seed)
    youngs_modulus_list = []
    coordinates_list = []
    for _ in range(num_samples):
        youngs_modulus = (
            min_youngs_modulus
            + torch.rand((1)) * (max_youngs_modulus - min_youngs_modulus)
        ).repeat(num_points, 1)
        youngs_modulus_list.append(youngs_modulus)
        coordinates = torch.rand((num_points, 1), requires_grad=True) * length
        coordinates_list.append(coordinates)
    return coordinates_list, youngs_modulus_list


def test_len(sut: StretchedRodSimulationDatasetLinearElasticity1D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_input_sample(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    expected_input_sample: tuple[list[Tensor], list[Tensor]],
    idx_sample: int,
) -> None:
    actual, _ = sut[idx_sample]

    x_coordinates_list, x_youngs_modulus_list = expected_input_sample
    x_coordinates = x_coordinates_list[idx_sample]
    x_youngs_modulus = x_youngs_modulus_list[idx_sample]
    expected = torch.concat((x_coordinates, x_youngs_modulus), dim=1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_output_sample(
    sut: StretchedRodSimulationDatasetLinearElasticity1D, idx_sample: int
) -> None:
    input, actual = sut[idx_sample]

    x_coordinates = input[:, 0].view((num_points, 1))
    x_youngs_modulus = input[:, 1].view((num_points, 1))
    expected = calculate_linear_elastic_displacements_solution(
        coordinates=x_coordinates,
        length=length,
        youngs_modulus=x_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    torch.testing.assert_close(actual, expected)


# ### Test collate_func()
@pytest.fixture
def fake_batch() -> list[tuple[Tensor, Tensor]]:
    sample_x_0 = torch.tensor([[1.0, 1.1]])
    sample_y_true_0 = torch.tensor([[2.0]])
    sample_x_1 = torch.tensor([[10.0, 10.1]])
    sample_y_true_1 = torch.tensor([[20.0]])
    return [(sample_x_0, sample_y_true_0), (sample_x_1, sample_y_true_1)]


def test_batch_pde__x(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    fake_batch: list[tuple[Tensor, Tensor]],
):
    collate_func = sut.get_collate_func()

    actual, _ = collate_func(fake_batch)

    expected = torch.tensor([[1.0, 1.1], [10.0, 10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_pde__y_true(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    fake_batch: list[tuple[Tensor, Tensor]],
):
    collate_func = sut.get_collate_func()

    _, actual = collate_func(fake_batch)

    expected = torch.tensor([[2.0], [20.0]])
    torch.testing.assert_close(actual, expected)
