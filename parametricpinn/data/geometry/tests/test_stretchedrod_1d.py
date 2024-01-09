import pytest
import torch

from parametricpinn.data.geometry import StretchedRod1D
from parametricpinn.settings import set_seed
from parametricpinn.types import Tensor

length = 1.0
num_points = 3
random_seed = 0
sobol_engine = torch.quasirandom.SobolEngine(dimension=1)


@pytest.fixture
def sut() -> StretchedRod1D:
    return StretchedRod1D(length=length)


def _generate_quasi_random_sobol_points() -> Tensor:
    min_coordinate = torch.tensor([0.0])
    normalized_lengths = sobol_engine.draw(num_points)
    return min_coordinate + (normalized_lengths * length)


def _generate_random_points() -> Tensor:
    return torch.rand((num_points, 1)) * length


def test_create_uniform_points(sut: StretchedRod1D) -> None:
    actual = sut.create_uniform_points(num_points=num_points)

    expected = torch.linspace(0.0, length, num_points).view((num_points, 1))
    torch.testing.assert_close(actual, expected)


def test_create_quasi_random_sobol_points(sut: StretchedRod1D) -> None:
    set_seed(random_seed)
    actual = sut.create_quasi_random_sobol_points(num_points=num_points)

    set_seed(random_seed)
    expected = _generate_quasi_random_sobol_points()
    torch.testing.assert_close(actual, expected)


def test_create_random_points(sut: StretchedRod1D) -> None:
    set_seed(random_seed)
    actual = sut.create_random_points(num_points=num_points)

    set_seed(random_seed)
    expected = _generate_random_points()
    torch.testing.assert_close(actual, expected)


def test_create_point_at_free_end(sut: StretchedRod1D) -> None:
    actual = sut.create_point_at_free_end()

    expected = torch.tensor([[length]])
    torch.testing.assert_close(actual, expected)
