import pytest
import torch

from parametricpinn.ansatz.hbc_ansatz_normalizers import (
    HBCAnsatzNormalizer,
    HBCAnsatzRenormalizer,
)
from parametricpinn.types import Tensor


# Test HBCAnsatzNormalizer
@pytest.mark.parametrize(
    ("min_values", "max_values", "values", "expected"),
    [
        (
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[0.0, 0.0], [5.0, 0.05], [10.0, 0.1]]),
            torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([[-10.0, -0.1], [-5.0, -0.05], [0.0, 0.0]]),
            torch.tensor([[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[-10.0, -0.1], [0.0, 0.0], [10.0, 0.1]]),
            torch.tensor([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]),
        ),
    ],
)
def test_hbc_ansatz_normalizer(
    min_values: Tensor, max_values: Tensor, values: Tensor, expected: Tensor
) -> None:
    sut = HBCAnsatzNormalizer(min_values=min_values, max_values=max_values)

    actual = sut(values)

    torch.testing.assert_close(actual, expected)


# Test HBCAnsatzRenormalizer
@pytest.mark.parametrize(
    ("min_values", "max_values", "normalied_values", "expected"),
    [
        (
            torch.tensor([0.0, 0.0]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            torch.tensor([[0.0, 0.0], [5.0, 0.05], [10.0, 0.1]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([[-1.0, -1.0], [-0.5, -0.5], [0.0, 0.0]]),
            torch.tensor([[-10.0, -0.1], [-5.0, -0.05], [0.0, 0.0]]),
        ),
        (
            torch.tensor([-10.0, -0.1]),
            torch.tensor([10.0, 0.1]),
            torch.tensor([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]),
            torch.tensor([[-10.0, -0.1], [0.0, 0.0], [10.0, 0.1]]),
        ),
    ],
)
def test_hbc_ansatz_renormalizer(
    min_values: Tensor, max_values: Tensor, normalied_values: Tensor, expected: Tensor
) -> None:
    sut = HBCAnsatzRenormalizer(min_values=min_values, max_values=max_values)

    actual = sut(normalied_values)

    torch.testing.assert_close(actual, expected)
