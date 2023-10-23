import pytest
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_hbc_ansatz_quarter_plate_with_hole,
)
from parametricpinn.network import FFNN
from parametricpinn.types import Tensor


class FakeNetwork(FFNN):
    def __init__(self) -> None:
        super().__init__(layer_sizes=[2, 2])

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 2), fill_value=2.0)


class FakeNetworkSingleInput(FFNN):
    def __init__(self) -> None:
        super().__init__(layer_sizes=[2, 2])

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([2.0, 2.0])


displacement_x_right = 0.0
displacement_y_bottom = 0.0
range_coordinates = torch.tensor([1.0, 1.0])


@pytest.fixture
def sut() -> StandardAnsatz:
    network = FakeNetwork()
    return create_standard_hbc_ansatz_quarter_plate_with_hole(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        range_coordinates=range_coordinates,
        network=network,
    )


def test_HBC_ansatz(sut: StandardAnsatz) -> None:
    inputs = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )

    actual = sut(inputs)

    expected = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ]
    )
    torch.testing.assert_close(actual, expected)


@pytest.fixture
def sut_single_input() -> StandardAnsatz:
    network = FakeNetworkSingleInput()
    return create_standard_hbc_ansatz_quarter_plate_with_hole(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        range_coordinates=range_coordinates,
        network=network,
    )


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0])),
        (torch.tensor([0.0, 0.5, 0.0, 0.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 2.0])),
        (torch.tensor([0.5, 0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0])),
        (torch.tensor([0.5, 0.5, 0.0, 0.0]), torch.tensor([1.0, 1.0])),
        (torch.tensor([0.5, 1.0, 0.0, 0.0]), torch.tensor([1.0, 2.0])),
        (torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([2.0, 0.0])),
        (torch.tensor([1.0, 0.5, 0.0, 0.0]), torch.tensor([2.0, 1.0])),
        (torch.tensor([1.0, 1.0, 0.0, 0.0]), torch.tensor([2.0, 2.0])),
    ],
)
def test_HBC_ansatz_for_single_input(
    sut_single_input: StandardAnsatz, input: Tensor, expected: Tensor
) -> None:
    actual = sut_single_input(input)

    torch.testing.assert_close(actual, expected)
