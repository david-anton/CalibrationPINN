import numpy as np
import pytest
import torch

from parametricpinn.calibration.likelihood import compile_likelihood
from parametricpinn.types import Tensor

device = torch.device("cpu")


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=1, keepdim=True)


def test_likelihood_func_single_data():
    model = FakeModel()
    coordinates = np.array([[1.0]])
    parameters = np.array([1.0])
    data = np.array([[1.0]])
    covariance_error = np.array([1 / 2])
    sut = compile_likelihood(
        model=model,
        coordinates=coordinates,
        data=data,
        covariance_error=covariance_error,
        device=device,
    )

    actual = sut(parameters)

    expected = (1 / np.sqrt(2 * np.pi * covariance_error[0])) * np.power(np.e, -1)
    assert actual == expected


def test_likelihood_func_multiple_data():
    model = FakeModel()
    coordinates = np.array([[1.0], [1.0]])
    parameters = np.array([1.0])
    data = np.array([[1.0], [1.0]])
    covariance_error = np.diag(np.full(2, 1 / 2))
    sut = compile_likelihood(
        model=model,
        coordinates=coordinates,
        data=data,
        covariance_error=covariance_error,
        device=device,
    )

    actual = sut(parameters)

    expected = (
        1 / (2 * np.pi * np.sqrt(np.linalg.det(covariance_error))) * np.power(np.e, -2)
    )
    assert actual == expected
