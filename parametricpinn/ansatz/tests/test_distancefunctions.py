import pytest
import torch

from parametricpinn.ansatz.distancefunctions import distance_function_factory
from parametricpinn.types import Tensor


def sigmoid_func(coordinate: Tensor, boundary_coordinate: Tensor) -> Tensor:
    relative_coordinate = coordinate - boundary_coordinate
    return (
        (2 * torch.exp(relative_coordinate)) / (torch.exp(relative_coordinate) + 1)
    ) - 1


@pytest.mark.parametrize(
    ("coordinates", "range_coordinate", "boundary_coordinate", "expected"),
    [
        (
            torch.tensor([[0.0], [5.0], [10.0]]),
            torch.tensor([10.0]),
            torch.tensor([0.0]),
            torch.tensor([[0.0], [0.5], [1.0]]),
        ),
        (
            torch.tensor([[-10.0], [-5.0], [0.0]]),
            torch.tensor([10.0]),
            torch.tensor([0.0]),
            torch.tensor([[-1.0], [-0.5], [0.0]]),
        ),
        (
            torch.tensor([[-5.0], [0.0], [5.0]]),
            torch.tensor([10.0]),
            torch.tensor([0.0]),
            torch.tensor([[-0.5], [0.0], [0.5]]),
        ),
        (
            torch.tensor([[10.0], [15.0], [20.0]]),
            torch.tensor([10.0]),
            torch.tensor([10.0]),
            torch.tensor([[0.0], [0.5], [1.0]]),
        ),
        (
            torch.tensor([[-20.0], [-15.0], [-10.0]]),
            torch.tensor([10.0]),
            torch.tensor([-10.0]),
            torch.tensor([[-1.0], [-0.5], [0.0]]),
        ),
        (
            torch.tensor([[5.0], [10.0], [15.0]]),
            torch.tensor([10.0]),
            torch.tensor([10.0]),
            torch.tensor([[-0.5], [0.0], [0.5]]),
        ),
        (
            torch.tensor([[-15.0], [-10.0], [-5.0]]),
            torch.tensor([10.0]),
            torch.tensor([-10.0]),
            torch.tensor([[-0.5], [0.0], [0.5]]),
        ),
    ],
)
def test_normalized_linear_distance_function(
    coordinates: Tensor,
    range_coordinate: Tensor,
    boundary_coordinate: Tensor,
    expected: Tensor,
) -> None:
    distance_function_type = "normalized linear"
    sut = distance_function_factory(
        type_str=distance_function_type,
        range_coordinate=range_coordinate,
        boundary_coordinate=boundary_coordinate,
    )

    actual = sut(coordinates)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("coordinates", "range_coordinate", "boundary_coordinate"),
    [
        (
            torch.tensor([[0.0], [5.0], [10.0]]),
            torch.tensor([10.0]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[-10.0], [-5.0], [0.0]]),
            torch.tensor([10.0]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[-5.0], [0.0], [5.0]]),
            torch.tensor([10.0]),
            torch.tensor([0.0]),
        ),
        (
            torch.tensor([[10.0], [15.0], [20.0]]),
            torch.tensor([10.0]),
            torch.tensor([10.0]),
        ),
        (
            torch.tensor([[-20.0], [-15.0], [-10.0]]),
            torch.tensor([10.0]),
            torch.tensor([-10.0]),
        ),
        (
            torch.tensor([[5.0], [10.0], [15.0]]),
            torch.tensor([10.0]),
            torch.tensor([10.0]),
        ),
        (
            torch.tensor([[-15.0], [-10.0], [-5.0]]),
            torch.tensor([10.0]),
            torch.tensor([-10.0]),
        ),
    ],
)
def test_sigmoid_distance_function(
    coordinates: Tensor, range_coordinate: Tensor, boundary_coordinate: Tensor
) -> None:
    distance_function_type = "sigmoid"
    expected = sigmoid_func(coordinates, boundary_coordinate)
    sut = distance_function_factory(
        type_str=distance_function_type,
        range_coordinate=range_coordinate,
        boundary_coordinate=boundary_coordinate,
    )

    actual = sut(coordinates)

    torch.testing.assert_close(actual, expected)
