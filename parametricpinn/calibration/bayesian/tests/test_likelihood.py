import math

import torch

from parametricpinn.calibration.bayesian.likelihood import compile_likelihood
from parametricpinn.calibration.data import PreprocessedData
from parametricpinn.types import Tensor

device = torch.device("cpu")


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sum(x, dim=1, keepdim=True)


def test_likelihood_func_single_data():
    model = FakeModel()
    inputs = torch.tensor([[1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0]])
    std_noise = torch.tensor([1 / torch.sqrt(torch.tensor(2))])
    covariance_error = std_noise**2
    data = PreprocessedData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        error_covariance_matrix=covariance_error,
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
        1 / torch.sqrt(2 * torch.tensor(math.pi) * covariance_error[0])
    ) * torch.pow(torch.tensor(math.e), -1)
    torch.testing.assert_close(actual, expected)


def test_likelihood_func_multiple_data():
    model = FakeModel()
    inputs = torch.tensor([[1.0], [1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0], [1.0]])
    std_noise = torch.diag(torch.full((2,), 1 / torch.sqrt(torch.tensor(2))))
    covariance_error = std_noise**2
    data = PreprocessedData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        error_covariance_matrix=covariance_error,
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
