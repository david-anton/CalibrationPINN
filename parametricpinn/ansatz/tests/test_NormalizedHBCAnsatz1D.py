# Standard library imports

# Third-party imports
import pytest
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.ansatz.normalizedHBCAnsatz1D import NormalizedHBCAnsatz1D
from parametricpinn.types import Tensor


class NetworkFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 1), fill_value=2.0)


class NormalizerFake(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


@pytest.fixture
def sut() -> NormalizedHBCAnsatz1D:
    input_range_coordinates = 1.0
    network = NetworkFake()
    input_normalizer = NormalizerFake()
    output_renormalizer = NormalizerFake()
    return NormalizedHBCAnsatz1D(
        network=network,
        input_normalizer=input_normalizer,
        output_renormalizer=output_renormalizer,
        input_range_coordinate=input_range_coordinates,
    )


def test_normalized_HBC_ansatz_1D(sut: NormalizedHBCAnsatz1D) -> None:
    inputs = torch.tensor([[0.0, 0.0], [0.5, 1.0], [1.0, 2.0]])
    actual = sut(inputs)

    expected = torch.tensor([[-1.0], [0.0], [1.0]])
    torch.testing.assert_close(actual, expected)
