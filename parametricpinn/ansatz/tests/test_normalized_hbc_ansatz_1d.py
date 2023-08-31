import pytest
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_1D,
)
from parametricpinn.network import FFNN
from parametricpinn.types import Tensor

displacement_left = torch.tensor([0.0])
min_inputs = torch.tensor([0.0, 0.0])
max_inputs = torch.tensor([2.0, 0.0])
min_outputs = torch.tensor([0.0])
max_outputs = torch.tensor([10.0])


class FakeNetwork(FFNN):
    def __init__(self) -> None:
        super().__init__(layer_sizes=[1, 1])

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 1), fill_value=1.0)


class FakeNetworkSingleInput(FFNN):
    def __init__(self) -> None:
        super().__init__(layer_sizes=[1, 1])

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([1.0])


@pytest.fixture
def sut() -> StandardAnsatz:
    network = FakeNetwork()
    return create_standard_normalized_hbc_ansatz_1D(
        displacement_left=displacement_left,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
        network=network,
    )


def test_normalized_HBC_ansatz_1D(sut: StandardAnsatz) -> None:
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    actual = sut(inputs)

    expected = torch.tensor([[0.0], [5.0], [10.0]])
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def sut_single_input() -> StandardAnsatz:
    network = FakeNetworkSingleInput()
    return create_standard_normalized_hbc_ansatz_1D(
        displacement_left=displacement_left,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
        network=network,
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
    sut_single_input: StandardAnsatz, input: Tensor, expected: Tensor
) -> None:
    actual = sut_single_input(input)

    torch.testing.assert_close(actual, expected)
