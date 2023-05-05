import pytest
import torch
import torch.nn as nn

from parametricpinn.ansatz import HBCAnsatz1D
from parametricpinn.types import Tensor


class FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 1), fill_value=2.0)


class FakeNetworkSingleInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([2.0])


range_coordinates = 1.0
displacement_left = 0.0


@pytest.fixture
def sut() -> HBCAnsatz1D:
    network = FakeNetwork()
    return HBCAnsatz1D(
        network=network,
        displacement_left=displacement_left,
        range_coordinate=range_coordinates,
    )


def test_HBC_ansatz_1D(sut: HBCAnsatz1D) -> None:
    inputs = torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])

    actual = sut(inputs)

    expected = torch.tensor([[0.0], [1.0], [2.0]])
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def sut_single_input() -> HBCAnsatz1D:
    network = FakeNetworkSingleInput()
    return HBCAnsatz1D(
        network=network,
        displacement_left=displacement_left,
        range_coordinate=range_coordinates,
    )


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (torch.tensor([0.0, 0.0]), torch.tensor([0.0])),
        (torch.tensor([0.5, 0.0]), torch.tensor([1.0])),
        (torch.tensor([1.0, 0.0]), torch.tensor([2.0])),
    ],
)
def test_HBC_ansatz_for_single_input(
    sut_single_input: HBCAnsatz1D, input: Tensor, expected: Tensor
) -> None:
    actual = sut_single_input(input)

    torch.testing.assert_close(actual, expected)