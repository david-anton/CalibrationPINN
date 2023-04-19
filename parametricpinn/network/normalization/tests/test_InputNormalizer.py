import pytest
import torch

from parametricpinn.network.normalization.inputnormlizer import InputNormalizer
from parametricpinn.types import Tensor


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
