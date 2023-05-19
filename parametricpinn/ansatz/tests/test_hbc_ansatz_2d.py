import pytest
import torch
import torch.nn as nn

from parametricpinn.ansatz import HBCAnsatz2D
from parametricpinn.types import Tensor


class FakeNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.full(size=(x.shape[0], 2), fill_value=2.0)


class FakeNetworkSingleInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.tensor([2.0, 2.0])


displacement_x_right = 0.0
displacement_y_bottom = 0.0
range_coordinate_x = 1.0
range_coordinate_y = 1.0


@pytest.fixture
def sut() -> HBCAnsatz2D:
    network = FakeNetwork()
    return HBCAnsatz2D(
        network=network,
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        range_coordinate_x=range_coordinate_x,
        range_coordinate_y=range_coordinate_y,
    )


def test_HBC_ansatz_2D(sut: HBCAnsatz2D) -> None:
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
def sut_single_input() -> HBCAnsatz2D:
    network = FakeNetworkSingleInput()
    return HBCAnsatz2D(
        network=network,
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        range_coordinate_x=range_coordinate_x,
        range_coordinate_y=range_coordinate_y,
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
    sut_single_input: HBCAnsatz2D, input: Tensor, expected: Tensor
) -> None:
    actual = sut_single_input(input)

    torch.testing.assert_close(actual, expected)
