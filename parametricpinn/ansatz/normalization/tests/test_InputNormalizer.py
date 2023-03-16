# Standard library imports

# Third-party imports
import pytest
import torch

# Local library imports
from parametricpinn.ansatz.normalization import InputNormalizer
from parametricpinn.types import Tensor


@pytest.mark.parametrize(
    ("min_inputs", "max_inputs", "inputs"),
    [
        (
            torch.Tensor([0.0, 0.0]),
            torch.Tensor([10.0, 0.1]),
            torch.Tensor([[0.0, 0.0], [5.0, 0.05], [10.0, 0.1]]),
        ),
        (
            torch.Tensor([-10.0, -0.1]),
            torch.Tensor([0.0, 0.0]),
            torch.Tensor([[-10.0, -0.1], [-5.0, -0.05], [0.0, 0.0]]),
        ),
        (
            torch.Tensor([-10.0, -0.1]),
            torch.Tensor([10.0, 0.1]),
            torch.Tensor([[-10.0, -0.1], [0.0, 0.0], [10.0, 0.1]]),
        ),
    ],
)
def test_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor, inputs: Tensor
) -> None:
    sut = InputNormalizer(min_inputs=min_inputs, max_inputs=max_inputs)

    actual = sut(inputs)

    expected = torch.Tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    torch.testing.assert_close(actual, expected)
