# Standard library imports

# Third-party imports
import pytest
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.ansatz.hbc_ansatz_1d import HBCAnsatz1D
from parametricpinn.types import Tensor


class FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 1), fill_value=2.0)


@pytest.fixture
def sut() -> HBCAnsatz1D:
    input_range_coordinates = 1.0
    network = FakeNetwork()
    return HBCAnsatz1D(
        network=network,
        displacement_left=0.0,
        input_range_coordinate=input_range_coordinates,
    )


def test_HBC_ansatz_1D(sut: HBCAnsatz1D) -> None:
    inputs = torch.tensor([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])

    actual = sut(inputs)

    expected = torch.tensor([[0.0], [1.0], [2.0]])
    torch.testing.assert_close(actual, expected)
