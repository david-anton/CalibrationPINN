import math

import torch

from parametricpinn.calibration.bayesian.likelihood import compile_likelihood
from parametricpinn.calibration.data import PreprocessedData
from parametricpinn.types import Tensor

device = torch.device("cpu")


class FakeModel_SingleDimension(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=1, keepdim=True)


class FakeModel_MultipleDimension(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.concat(
            (torch.sum(x, dim=1, keepdim=True), torch.sum(x, dim=1, keepdim=True)),
            dim=1,
        )


def test_likelihood_func_single_data_single_dimension():
    model = FakeModel_SingleDimension()
    inputs = torch.tensor([[1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = std_noise**2
    data = PreprocessedData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=1,
    )
    sut = compile_likelihood(
        model=model,
        data=data,
        device=device,
    )

    actual = sut(parameters)

    expected = (
        1 / torch.sqrt(2 * torch.tensor(math.pi) * covariance_error)
    ) * torch.pow(torch.tensor(math.e), -1)
    torch.testing.assert_close(actual, expected)


def test_likelihood_func_multiple_data_single_dimension():
    model = FakeModel_SingleDimension()
    inputs = torch.tensor([[1.0], [1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0], [1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((2,), std_noise**2))
    data = PreprocessedData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=1,
    )
    sut = compile_likelihood(
        model=model,
        data=data,
        device=device,
    )

    actual = sut(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(covariance_error)))
        * torch.pow(torch.tensor(math.e), -2)
    )
    torch.testing.assert_close(actual, expected)


def test_likelihood_func_single_data_multiple_dimension():
    model = FakeModel_MultipleDimension()
    inputs = torch.tensor([[0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((2,), std_noise**2))
    data = PreprocessedData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=2,
    )
    sut = compile_likelihood(
        model=model,
        data=data,
        device=device,
    )

    actual = sut(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(covariance_error)))
        * torch.pow(torch.tensor(math.e), -2)
    )
    torch.testing.assert_close(actual, expected)


def test_likelihood_func_multiple_data_multiple_dimension():
    model = FakeModel_MultipleDimension()
    inputs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((4,), std_noise**2))
    data = PreprocessedData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=2,
    )
    sut = compile_likelihood(
        model=model,
        data=data,
        device=device,
    )

    actual = sut(parameters)

    expected = (
        1
        / (torch.pow((2 * torch.tensor(math.pi)), 2) * torch.sqrt(torch.det(covariance_error)))
        * torch.pow(torch.tensor(math.e), -4)
    )
    torch.testing.assert_close(actual, expected)
