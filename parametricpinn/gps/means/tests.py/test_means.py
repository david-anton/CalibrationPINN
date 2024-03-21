import pytest
import torch

from parametricpinn.gps.means import ConstantMean, ZeroMean
from parametricpinn.types import Tensor

device = torch.device("cpu")
constant_mean = 2.0


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (torch.tensor([1.0]), torch.tensor([0.0])),
        (torch.tensor([0.0]), torch.tensor([0.0])),
        (torch.tensor([-1.0]), torch.tensor([0.0])),
        (torch.tensor([-1.0, 0.0, 1.0]), torch.tensor([0.0, 0.0, 0.0])),
    ],
)
def test_zero_mean(inputs: Tensor, expected: Tensor) -> None:
    sut = ZeroMean(device)
    actual = sut(inputs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (torch.tensor([1.0]), torch.tensor([constant_mean])),
        (torch.tensor([0.0]), torch.tensor([constant_mean])),
        (torch.tensor([-1.0]), torch.tensor([constant_mean])),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            torch.tensor([constant_mean, constant_mean, constant_mean]),
        ),
    ],
)
def test_constant_mean(inputs: Tensor, expected: Tensor) -> None:
    sut = ConstantMean(device)
    sut.set_parameters(torch.tensor([constant_mean], device=device))
    actual = sut(inputs)

    torch.testing.assert_close(actual, expected)
