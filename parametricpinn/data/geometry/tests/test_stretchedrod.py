# Standard library imports

# Third-party imports
import pytest
import torch

# Local library imports
from parametricpinn.data.geometry import StretchedRod
from parametricpinn.settings import set_seed

length = 1.0
num_points = 3
random_seed = 0


@pytest.fixture
def sut() -> StretchedRod:
    return StretchedRod(length=length)


def test_create_uniform_points(sut: StretchedRod) -> None:
    actual = sut.create_uniform_points(num_points=num_points)

    expected = torch.linspace(0.0, length, num_points).view((num_points, 1))
    torch.testing.assert_close(actual, expected)


def test_create_random_points(sut: StretchedRod) -> None:
    set_seed(random_seed)
    actual = sut.create_random_points(num_points=num_points)

    set_seed(random_seed)
    expected = torch.rand((num_points, 1)) * length
    torch.testing.assert_close(actual, expected)


def test_create_point_at_free_end(sut: StretchedRod) -> None:
    actual = sut.create_point_at_free_end()

    expected = torch.tensor([[length]])
    torch.testing.assert_close(actual, expected)
