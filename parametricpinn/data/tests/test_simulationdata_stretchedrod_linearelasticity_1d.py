import pytest
import torch

from parametricpinn.data.dataset import SimulationData, SimulationDataList
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


### Test SimulationDataset
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
def expected_x() -> tuple[list[Tensor], list[Tensor]]:
    # The random numbers must be generated in the same order as in the system under test.
    set_seed(random_seed)
    coordinates_list = []
    youngs_modulus_list = []
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
def test_x_coor(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    expected_x: tuple[list[Tensor], list[Tensor]],
    idx_sample: int,
) -> None:
    sample = sut[idx_sample]
    actual = sample.x_coor

    x_coor_list, _ = expected_x
    expected = x_coor_list[idx_sample]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_x_params(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    expected_x: tuple[list[Tensor], list[Tensor]],
    idx_sample: int,
) -> None:
    sample = sut[idx_sample]
    actual = sample.x_params

    _, x_params_list = expected_x
    expected = x_params_list[idx_sample]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_y_true(
    sut: StretchedRodSimulationDatasetLinearElasticity1D, idx_sample: int
) -> None:
    sample = sut[idx_sample]
    actual = sample.y_true

    x_coordinates = sample.x_coor
    x_youngs_modulus = sample.x_params
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
def fake_batch() -> SimulationDataList:
    sample_0 = SimulationData(
        x_coor=torch.tensor([[1.0, 1.1]]),
        x_params=torch.tensor([[2.0, 2.1]]),
        y_true=torch.tensor([[3.0]]),
    )
    sample_1 = SimulationData(
        x_coor=torch.tensor([[10.0, 10.1]]),
        x_params=torch.tensor([[20.0, 20.1]]),
        y_true=torch.tensor([[30.0]]),
    )
    return [sample_0, sample_1]


def test_batch_x_coor(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    fake_batch: SimulationDataList,
):
    collate_func = sut.get_collate_func()

    batched_data = collate_func(fake_batch)
    actual = batched_data.x_coor

    expected = torch.tensor([[1.0, 1.1], [10.0, 10.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_x_params(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    fake_batch: SimulationDataList,
):
    collate_func = sut.get_collate_func()

    batched_data = collate_func(fake_batch)
    actual = batched_data.x_params

    expected = torch.tensor([[2.0, 2.1], [20.0, 20.1]])
    torch.testing.assert_close(actual, expected)


def test_batch_y_true(
    sut: StretchedRodSimulationDatasetLinearElasticity1D,
    fake_batch: SimulationDataList,
):
    collate_func = sut.get_collate_func()

    batched_data = collate_func(fake_batch)
    actual = batched_data.y_true

    expected = torch.tensor([[3.0], [30.0]])
    torch.testing.assert_close(actual, expected)
