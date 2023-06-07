import pytest
import torch
import torch.nn as nn

from parametricpinn.ansatz import NormalizedHBCAnsatz2D, create_normalized_hbc_ansatz_2D
from parametricpinn.types import Tensor

displacement_x_right = torch.tensor([0.0])
displacement_y_bottom = torch.tensor([0.0])
min_inputs = torch.tensor([-2.0, 0.0, 0.0, 0.0])
max_inputs = torch.tensor([0.0, 2.0, 0.0, 0.0])
min_outputs = torch.tensor([0.0, 0.0])
max_outputs = torch.tensor([10.0, 10.0])


class FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 2), fill_value=1.0)


class FakeNetworkSingleInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([1.0, 1.0])


@pytest.fixture
def sut() -> NormalizedHBCAnsatz2D:
    network = FakeNetwork()
    return create_normalized_hbc_ansatz_2D(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        network=network,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
    )


def test_normalized_HBC_ansatz_2D(sut: NormalizedHBCAnsatz2D) -> None:
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
            [-10.0, 10.0],
            [-5.0, 10.0],
            [0.0, 10.0],
            [-10.0, 5.0],
            [-5.0, 5.0],
            [0.0, 5.0],
            [-10.0, 0.0],
            [-5.0, 0.0],
            [0.0, 0.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def sut_single_input() -> NormalizedHBCAnsatz2D:
    network = FakeNetworkSingleInput()
    return create_normalized_hbc_ansatz_2D(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        network=network,
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
    )


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (torch.tensor([-2.0, 2.0, 0.0, 0.0]), torch.tensor([-10.0, 10.0])),
        (torch.tensor([-1.0, 2.0, 0.0, 0.0]), torch.tensor([-5.0, 10.0])),
        (torch.tensor([0.0, 2.0, 0.0, 0.0]), torch.tensor([0.0, 10.0])),
        (torch.tensor([-2.0, 1.0, 0.0, 0.0]), torch.tensor([-10.0, 5.0])),
        (torch.tensor([-1.0, 1.0, 0.0, 0.0]), torch.tensor([-5.0, 5.0])),
        (torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 5.0])),
        (torch.tensor([-2.0, 0.0, 0.0, 0.0]), torch.tensor([-10.0, 0.0])),
        (torch.tensor([-1.0, 0.0, 0.0, 0.0]), torch.tensor([-5.0, 0.0])),
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])),
    ],
)
def test_normalized_HBC_ansatz_2D_for_single_input(
    sut_single_input: NormalizedHBCAnsatz2D, input: Tensor, expected: Tensor
) -> None:
    actual = sut_single_input(input)

    torch.testing.assert_close(actual, expected)
