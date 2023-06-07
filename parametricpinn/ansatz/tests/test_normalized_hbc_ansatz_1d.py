import pytest
import torch
import torch.nn as nn

from parametricpinn.ansatz import NormalizedHBCAnsatz1D, create_normalized_hbc_ansatz_1D
from parametricpinn.types import Tensor

displacement_left = torch.tensor([0.0])
min_inputs = torch.tensor([0.0, 0.0])
max_inputs = torch.tensor([2.0, 0.0])
min_outputs = torch.tensor([0.0])
max_outputs = torch.tensor([10.0])


class FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 1), fill_value=1.0)


class FakeNetworkSingleInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([1.0])


@pytest.fixture
def sut() -> NormalizedHBCAnsatz1D:
    network = FakeNetwork()
    return create_normalized_hbc_ansatz_1D(
        displacement_left=displacement_left,
        network=network,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
    )


def test_normalized_HBC_ansatz_1D(sut: NormalizedHBCAnsatz1D) -> None:
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    actual = sut(inputs)

    expected = torch.tensor([[0.0], [5.0], [10.0]])
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def sut_single_input() -> NormalizedHBCAnsatz1D:
    network = FakeNetworkSingleInput()
    return create_normalized_hbc_ansatz_1D(
        displacement_left=displacement_left,
        network=network,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
    )


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (torch.tensor([0.0, 0.0]), torch.tensor([0.0])),
        (torch.tensor([1.0, 0.0]), torch.tensor([5.0])),
        (torch.tensor([2.0, 0.0]), torch.tensor([10.0])),
    ],
)
def test_normalized_HBC_ansatz_1D_for_single_input(
    sut_single_input: NormalizedHBCAnsatz1D, input: Tensor, expected: Tensor
) -> None:
    actual = sut_single_input(input)

    torch.testing.assert_close(actual, expected)
