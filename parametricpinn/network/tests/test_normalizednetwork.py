import pytest
import torch
import torch.nn as nn

from parametricpinn.network import create_normalized_network
from parametricpinn.network.normalizednetwork import (
    InputNormalizer,
    NormalizedNetwork,
    OutputRenormalizer,
)
from parametricpinn.settings import set_default_dtype
from parametricpinn.types import Tensor

set_default_dtype(torch.float64)

absolute_tolerance = torch.tensor([1e-7])


# Test InputNormalizer
@pytest.mark.parametrize(
    ("min_inputs", "max_inputs", "inputs"),
    [
        (
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[0.0, 0.0], [5.0, 0.05], [10.0, 0.1]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([[-10.0, -0.1], [-5.0, -0.05], [0.0, 0.0]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[-10.0, -0.1], [0.0, 0.0], [10.0, 0.1]]),
        ),
    ],
)
def test_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor, inputs: Tensor
) -> None:
    sut = InputNormalizer(min_inputs=min_inputs, max_inputs=max_inputs)

    actual = sut(inputs)

    expected = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("min_inputs", "max_inputs", "inputs"),
    [
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([-10.0, -0.1]),
            torch.tensor([-10.0, -0.1]),
        ),
        (
            torch.tensor([10.0, 0.1]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([10.0, 0.1]),
        ),
        (
            torch.tensor([-10.0, 10.0]),
            torch.tensor([-10.0, 10.0]),
            torch.tensor([-10.0, 10.0]),
        ),
        (
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        ),
    ],
)
def test_input_normalizer_for_one_input(
    min_inputs: Tensor, max_inputs: Tensor, inputs: Tensor
) -> None:
    sut = InputNormalizer(min_inputs=min_inputs, max_inputs=max_inputs)

    actual = sut(inputs)

    expected = torch.tensor([0.0, 0.0])
    torch.testing.assert_close(actual, expected)


# Test OutputRenormalizer
@pytest.mark.parametrize(
    ("min_outputs", "max_outputs", "expected"),
    [
        (
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[0.0, 0.0], [5.0, 0.05], [10.0, 0.1]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([[-10.0, -0.1], [-5.0, -0.05], [0.0, 0.0]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[-10.0, -0.1], [0.0, 0.0], [10.0, 0.1]]),
        ),
    ],
)
def test_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor, expected: Tensor
) -> None:
    sut = OutputRenormalizer(min_outputs=min_outputs, max_outputs=max_outputs)

    output = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    actual = sut(output)

    torch.testing.assert_close(actual, expected)


# Test NormalizedNetwork
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
