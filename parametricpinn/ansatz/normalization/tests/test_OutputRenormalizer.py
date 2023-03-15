# Standard library imports

# Third-party imports
import pytest
import torch

# Local library imports
from parametricpinn.ansatz.normalization import OutputRenormalizer


@pytest.mark.parametrize(
    ("min_outputs", "max_outputs", "expected"),
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
def test_output_renormalizer(min_outputs, max_outputs, expected):
    sut = OutputRenormalizer(min_outputs=min_outputs, max_outputs=max_outputs)

    output = torch.Tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    actual = sut(output)

    torch.testing.assert_close(actual, expected)
