import pytest
import torch

from calibrationpinn.training.metrics import (
    l2_norm,
    mean_absolute_error,
    mean_absolute_relative_error,
    mean_relative_error,
    mean_squared_error,
    relative_l2_norm,
)
from calibrationpinn.types import Tensor


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            torch.tensor([[1.0], [1.0], [1.0]]),
            torch.tensor([[3.0], [0.0], [2.0]]),
            torch.tensor(2.0),
        ),
        (
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[3.0, 3.0], [0.0, 0.0], [2.0, 2.0]]),
            torch.tensor(2.0),
        ),
    ],
)
def test_mean_squared_error(y_true: Tensor, y_pred: Tensor, expected: Tensor) -> None:
    sut = mean_squared_error

    actual = sut(y_true=y_true, y_pred=y_pred)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            torch.tensor([[1.0], [1.0], [1.0]]),
            torch.tensor([[3.0], [0.5], [1.5]]),
            torch.tensor(1.0),
        ),
        (
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[3.0, 3.0], [0.5, 0.5], [1.5, 1.5]]),
            torch.tensor(1.0),
        ),
    ],
)
def test_mean_absolute_error(y_true: Tensor, y_pred: Tensor, expected: Tensor) -> None:
    sut = mean_absolute_error

    actual = sut(y_true=y_true, y_pred=y_pred)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            torch.tensor([[1.0], [1.0], [1.0]]),
            torch.tensor([[4.0], [0.0], [2.0]]),
            torch.tensor(1.0),
        ),
        (
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[4.0, 4.0], [0.0, 0.0], [2.0, 2.0]]),
            torch.tensor(1.0),
        ),
    ],
)
def test_mean_relative_error(y_true: Tensor, y_pred: Tensor, expected: Tensor) -> None:
    sut = mean_relative_error

    actual = sut(y_true=y_true, y_pred=y_pred)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            torch.tensor([[1.0], [1.0], [1.0]]),
            torch.tensor([[5.0], [0.0], [2.0]]),
            torch.tensor(2.0),
        ),
        (
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[5.0, 5.0], [0.0, 0.0], [2.0, 2.0]]),
            torch.tensor(2.0),
        ),
    ],
)
def test_mean_absolute_relative_error(
    y_true: Tensor, y_pred: Tensor, expected: Tensor
) -> None:
    sut = mean_absolute_relative_error

    actual = sut(y_true=y_true, y_pred=y_pred)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            torch.tensor([[1.0], [1.0], [1.0]]),
            torch.tensor([[3.0], [0.0], [2.0]]),
            torch.sqrt(torch.tensor(2.0) ** 2 + 2 * (torch.tensor(1.0) ** 2)),
        ),
        (
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[3.0, 3.0], [0.0, 0.0], [2.0, 2.0]]),
            torch.sqrt(2 * (torch.tensor(2.0) ** 2) + 4 * (torch.tensor(1.0) ** 2)),
        ),
    ],
)
def test_l2_norm(y_true: Tensor, y_pred: Tensor, expected: Tensor) -> None:
    sut = l2_norm

    actual = sut(y_true=y_true, y_pred=y_pred)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (
            torch.tensor([[1.0], [1.0], [1.0]]),
            torch.tensor([[3.0], [0.0], [2.0]]),
            torch.sqrt(torch.tensor(2.0) ** 2 + 2 * (torch.tensor(1.0) ** 2))
            / torch.sqrt(3 * (torch.tensor(1.0) ** 2)),
        ),
        (
            torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
            torch.tensor([[3.0, 3.0], [0.0, 0.0], [2.0, 2.0]]),
            torch.sqrt(2 * (torch.tensor(2.0) ** 2) + 4 * (torch.tensor(1.0) ** 2))
            / torch.sqrt(6 * (torch.tensor(1.0) ** 2)),
        ),
    ],
)
def test_relative_l2_norm(y_true: Tensor, y_pred: Tensor, expected: Tensor) -> None:
    sut = relative_l2_norm

    actual = sut(y_true=y_true, y_pred=y_pred)

    torch.testing.assert_close(actual, expected)
