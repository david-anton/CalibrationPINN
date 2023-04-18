# Standard library imports

# Third-party imports
import pytest
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.network import create_normalized_network
from parametricpinn.network.normalization.normalizednetwork import NormalizedNetwork
from parametricpinn.types import Tensor


class FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


min_inputs = torch.tensor([-10.0, -20.0])
max_inputs = torch.tensor([10.0, 20.0])
min_outputs = torch.tensor([-100.0, -200.0])
max_outputs = torch.tensor([100.0, 200.0])


@pytest.fixture
def sut() -> NormalizedNetwork:
    return create_normalized_network(
        network=FakeNetwork(),
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
    )


def test_normalized_network(sut: NormalizedNetwork) -> None:
    inputs = torch.vstack((min_inputs, max_inputs))

    actual = sut(inputs)

    expected = torch.vstack((min_outputs, max_outputs))
    torch.testing.assert_close(actual, expected)
