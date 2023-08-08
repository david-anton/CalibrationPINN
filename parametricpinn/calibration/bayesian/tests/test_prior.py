import pytest
import torch

from parametricpinn.calibration.bayesian.prior import (
    compile_mixed_multivariate_independently_distributed_prior,
    compile_multivariate_normal_distributed_prior,
    compile_univariate_normal_distributed_prior,
    compile_univariate_uniform_distributed_prior,
)
from parametricpinn.types import Tensor

device = torch.device("cpu")


def _expected_univariate_normal_distributed_prior() -> list[tuple[Tensor, Tensor]]:
    mean = 0.0
    standard_deviation = 1.0
    return [
        (
            torch.tensor([0.0]),
            torch.exp(
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([0.0])
                )
            ),
        ),
        (
            torch.tensor([-1.0]),
            torch.exp(
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([-1.0])
                )
            ),
        ),
        (
            torch.tensor([1.0]),
            torch.exp(
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([1.0])
                )
            ),
        ),
    ]


def _expected_multivariate_normal_distributed_prior() -> list[tuple[Tensor, Tensor]]:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return [
        (
            torch.tensor([0.0, 0.0]),
            torch.exp(
                torch.distributions.MultivariateNormal(
                    loc=means, covariance_matrix=covariance_matrix
                ).log_prob(torch.tensor([0.0, 0.0]))
            ),
        ),
        (
            torch.tensor([-1.0, -1.0]),
            torch.exp(
                torch.distributions.MultivariateNormal(
                    loc=means, covariance_matrix=covariance_matrix
                ).log_prob(torch.tensor([-1.0, -1.0]))
            ),
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.exp(
                torch.distributions.MultivariateNormal(
                    loc=means, covariance_matrix=covariance_matrix
                ).log_prob(torch.tensor([1.0, 1.0]))
            ),
        ),
    ]


def _expected_mixed_multivariate_independently_distributed_prior() -> (
    list[tuple[Tensor, Tensor]]
):
    mean_normal_dist = 0.0
    standard_deviation_normal_dist = 1.0
    parameter_normal_dist = torch.tensor(0.0)
    probability_normal_dist = torch.exp(
        torch.distributions.Normal(
            loc=mean_normal_dist, scale=standard_deviation_normal_dist
        ).log_prob(parameter_normal_dist)
    )

    return [
        (torch.tensor([0.0, parameter_normal_dist]), 0.5 * probability_normal_dist),
        (torch.tensor([-2.0, parameter_normal_dist]), 0.0 * probability_normal_dist),
        (torch.tensor([2.0, parameter_normal_dist]), 0.0 * probability_normal_dist),
    ]


@pytest.mark.parametrize(
    ("parameter", "expected"),
    [
        (torch.tensor([0.0]), torch.tensor([0.5])),
        (torch.tensor([-2.0]), torch.tensor([0.0])),
        (torch.tensor([2.0]), torch.tensor([0.0])),
    ],
)
def test_univariate_uniform_distributed_prior(parameter: Tensor, expected: Tensor):
    lower_limit = -1.0
    upper_limit = 1.0
    sut = compile_univariate_uniform_distributed_prior(
        lower_limit=lower_limit, upper_limit=upper_limit, device=device
    )

    actual = sut(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_normal_distributed_prior()
)
def test_univariate_normal_distributed_prior(parameter: Tensor, expected: Tensor):
    mean = 0.0
    standard_deviation = 1.0
    sut = compile_univariate_normal_distributed_prior(
        mean=mean, standard_deviation=standard_deviation, device=device
    )

    actual = sut(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_normal_distributed_prior()
)
def test_multivariate_normal_distributed_prior(parameter: Tensor, expected: Tensor):
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sut = compile_multivariate_normal_distributed_prior(
        means=means, covariance_matrix=covariance_matrix, device=device
    )

    actual = sut(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_mixed_multivariate_independently_distributed_prior(),
)
def test_mixed_multivariate_independently_distributed_prior(
    parameter: Tensor, expected: Tensor
):
    lower_bound_uniform_dist = -1
    upper_bound_uniform_dist = 1
    uniform_dist = torch.distributions.Uniform(
        low=lower_bound_uniform_dist, high=upper_bound_uniform_dist, validate_args=False
    )
    mean_normal_dist = 0.0
    standard_deviation_normal_dist = 1.0
    normal_dist = torch.distributions.Normal(
        loc=mean_normal_dist, scale=standard_deviation_normal_dist
    )
    sut = compile_mixed_multivariate_independently_distributed_prior(
        independent_univariate_distributions=[uniform_dist, normal_dist]
    )

    actual = sut(parameter)

    torch.testing.assert_close(actual, expected)
