import pytest
import torch

from parametricpinn.gps.normalizers import InputNormalizer
from parametricpinn.settings import set_default_dtype
from parametricpinn.types import Tensor

set_default_dtype(torch.float64)

device = torch.device("cpu")
absolute_tolerance = torch.tensor([1e-7])


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
    sut = InputNormalizer(min_inputs=min_inputs, max_inputs=max_inputs, device=device)

    actual = sut(inputs)

    expected = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
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
    sut = InputNormalizer(min_inputs=min_inputs, max_inputs=max_inputs, device=device)

    actual = sut(inputs)

    expected = torch.tensor([0.0, 0.0])
    torch.testing.assert_close(actual, expected)
