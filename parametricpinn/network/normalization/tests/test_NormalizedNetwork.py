# Standard library imports

# Third-party imports
import pytest
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.network import NormalizedNetwork
from parametricpinn.network.normalization.outputRenormalizer import OutputRenormalizer
from parametricpinn.network.normalization.inputNormlizer import InputNormalizer
from parametricpinn.types import Tensor


class NetworkFake(nn.Module):
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
    network = NetworkFake()
    input_normalizer = InputNormalizer(min_inputs=min_inputs, max_inputs=max_inputs)
    output_renormalizer = OutputRenormalizer(
        min_outputs=min_outputs, max_outputs=max_outputs
    )
    return NormalizedNetwork(
        network=network,
        input_normalizer=input_normalizer,
        output_renormalizer=output_renormalizer,
    )


def test_normalized_network(sut: NormalizedNetwork) -> None:
    inputs = torch.vstack((min_inputs, max_inputs))

    actual = sut(inputs)

    expected = torch.vstack((min_outputs, max_outputs))
    torch.testing.assert_close(actual, expected)
