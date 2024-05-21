import pytest
import torch

from calibrationpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_quarter_plate_with_hole,
)
from calibrationpinn.network import FFNN
from calibrationpinn.settings import set_default_dtype
from calibrationpinn.types import Tensor

set_default_dtype(torch.float64)
device = torch.device("cpu")

displacement_x_right = torch.tensor([0.0])
displacement_y_bottom = torch.tensor([0.0])
min_inputs = torch.tensor([-2.0, 0.0, 0.0, 0.0])
max_inputs = torch.tensor([0.0, 2.0, 0.0, 0.0])
min_outputs = torch.tensor([-10.0, -20.0])
max_outputs = torch.tensor([0.0, 0.0])
distance_function_type = "normalized linear"


class FakeNetwork(FFNN):
    def __init__(self) -> None:
        super().__init__(layer_sizes=[2, 2])

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([2.0, -2.0]).repeat(x.shape[0], 1)


class FakeNetworkSingleInput(FFNN):
    def __init__(self) -> None:
        super().__init__(layer_sizes=[2, 2])

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([2.0, -2.0])


@pytest.fixture
def sut() -> StandardAnsatz:
    network = FakeNetwork()
    return create_standard_normalized_hbc_ansatz_quarter_plate_with_hole(
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
        network=network,
        distance_function_type=distance_function_type,
        device=device,
    )


def test_normalized_HBC_ansatz(sut: StandardAnsatz) -> None:
    inputs = torch.tensor(
        [
            [-2.0, 2.0, 0.0, 0.0],
            [-1.0, 2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [-2.0, 1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    actual = sut(inputs)

    expected = torch.tensor(
        [
            [-10.0, -20.0],
            [-5.0, -20.0],
            [0.0, -20.0],
            [-10.0, -10.0],
            [-5.0, -10.0],
            [0.0, -10.0],
            [-10.0, 0.0],
            [-5.0, 0.0],
            [0.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def sut_single_input() -> StandardAnsatz:
    network = FakeNetworkSingleInput()
    return create_standard_normalized_hbc_ansatz_quarter_plate_with_hole(
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
        network=network,
        distance_function_type=distance_function_type,
        device=device,
    )


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (torch.tensor([-2.0, 2.0, 0.0, 0.0]), torch.tensor([-10.0, -20.0])),
        (torch.tensor([-1.0, 2.0, 0.0, 0.0]), torch.tensor([-5.0, -20.0])),
        (torch.tensor([0.0, 2.0, 0.0, 0.0]), torch.tensor([0.0, -20.0])),
        (torch.tensor([-2.0, 1.0, 0.0, 0.0]), torch.tensor([-10.0, -10.0])),
        (torch.tensor([-1.0, 1.0, 0.0, 0.0]), torch.tensor([-5.0, -10.0])),
        (torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, -10.0])),
        (torch.tensor([-2.0, 0.0, 0.0, 0.0]), torch.tensor([-10.0, 0.0])),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), torch.tensor([-5.0, 0.0])),
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])),
    ],
)
def test_normalized_HBC_ansatz_for_single_input(
    sut_single_input: StandardAnsatz, input: Tensor, expected: Tensor
) -> None:
    actual = sut_single_input(input)

    torch.testing.assert_close(actual, expected)
