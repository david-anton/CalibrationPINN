# Standard library imports

# Third-party imports
import torch

# Local library imports
from parametricpinn.types import Tensor


def mean_squared_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return torch.mean(torch.square(y_pred - y_true))


def mean_absolute_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return torch.mean(torch.abs(y_pred - y_true))


def mean_relative_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return torch.mean((y_pred - y_true) / y_true)


def mean_absolute_relative_error(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return torch.mean(torch.abs((y_pred - y_true) / y_true))


def l2_norm(y_true: Tensor, y_pred: Tensor) -> Tensor:
    return torch.sqrt(torch.sum(torch.square(y_pred - y_true)))


def relative_l2_norm(y_true: Tensor, y_pred: Tensor) -> Tensor:
    def l2_norm(array: Tensor) -> Tensor:
        return torch.sqrt(torch.sum(torch.square(array)))

    return torch.divide(l2_norm(y_pred - y_true), l2_norm(y_true))
