import pytest
import torch

from parametricpinn.ansatz.distancefunctions import distance_function_factory
from parametricpinn.types import Tensor


def sigmoid_func(x: Tensor) -> Tensor:
    return ((2 * torch.exp(x)) / (torch.exp(x) + 1)) - 1


@pytest.mark.parametrize(
    ("coordinates", "range_coordinate", "expected"),
    [
        (
            torch.tensor([[0.0], [5.0], [10.0]]),
            torch.tensor([10.0]),
            torch.tensor([[0.0], [0.5], [1.0]]),
        ),
        (
            torch.tensor([[-10.0], [-5.0], [0.0]]),
            torch.tensor([10.0]),
            torch.tensor([[-1.0], [-0.5], [0.0]]),
        ),
        (
            torch.tensor([[-5.0], [0.0], [5.0]]),
            torch.tensor([10.0]),
            torch.tensor([[-0.5], [0.0], [0.5]]),
        ),
    ],
)
def test_normalized_linear_distance_function(
    coordinates: Tensor, range_coordinate: Tensor, expected: Tensor
) -> None:
    distance_function_type = "normalized linear"
    sut = distance_function_factory(
        type_str=distance_function_type, range_coordinate=range_coordinate
    )

    actual = sut(coordinates)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("coordinates", "range_coordinate"),
    [
        (torch.tensor([[0.0], [5.0], [10.0]]), torch.tensor([10.0])),
        (torch.tensor([[-10.0], [-5.0], [0.0]]), torch.tensor([10.0])),
        (torch.tensor([[-5.0], [0.0], [5.0]]), torch.tensor([10.0])),
    ],
)
def test_sigmoid_distance_function(
    coordinates: Tensor, range_coordinate: Tensor
) -> None:
    distance_function_type = "sigmoid"
    expected = sigmoid_func(coordinates)
    sut = distance_function_factory(
        type_str=distance_function_type, range_coordinate=range_coordinate
    )

    actual = sut(coordinates)

    torch.testing.assert_close(actual, expected)
