# Standard library imports

# Third-party imports
import pytest
import torch

# Local library imports
from parametricpinn.network.normalization.outputRenormalizer import OutputRenormalizer
from parametricpinn.types import Tensor


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
