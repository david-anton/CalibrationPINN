import pytest
import torch

from parametricpinn.calibration.bayesian.distributions import (
    create_independent_multivariate_normal_distribution,
    create_mixed_multivariate_independently_distribution,
    create_multivariate_normal_distribution,
    create_univariate_normal_distribution,
    create_univariate_uniform_distribution,
)
from parametricpinn.types import Tensor

device = torch.device("cpu")


def _expected_univariate_uniform_distribution() -> list[tuple[Tensor, Tensor]]:
    return [
        (torch.tensor([0.0]), torch.log(torch.tensor(0.5, dtype=torch.float64))),
        (torch.tensor([-2.0]), torch.log(torch.tensor(0.0, dtype=torch.float64))),
        (torch.tensor([2.0]), torch.log(torch.tensor(0.0, dtype=torch.float64))),
    ]


def _expected_univariate_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    mean = torch.tensor(0.0)
    standard_deviation = torch.tensor(1.0)
    return [
        (
            torch.tensor([0.0]),
            torch.distributions.Normal(loc=mean, scale=standard_deviation)
            .log_prob(torch.tensor([0.0]))
            .type(torch.float64)[0],
        ),
        (
            torch.tensor([-1.0]),
            torch.distributions.Normal(loc=mean, scale=standard_deviation)
            .log_prob(torch.tensor([-1.0]))
            .type(torch.float64)[0],
        ),
        (
            torch.tensor([1.0]),
            torch.distributions.Normal(loc=mean, scale=standard_deviation)
            .log_prob(torch.tensor([1.0]))
            .type(torch.float64)[0],
        ),
    ]


def _expected_multivariate_normal_distribution() -> list[tuple[Tensor, Tensor]]:
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    return [
        (
            torch.tensor([0.0, 0.0]),
            torch.distributions.MultivariateNormal(
                loc=means, covariance_matrix=covariance_matrix
            )
            .log_prob(torch.tensor([0.0, 0.0]))
            .type(torch.float64),
        ),
        (
            torch.tensor([-1.0, -1.0]),
            torch.distributions.MultivariateNormal(
                loc=means, covariance_matrix=covariance_matrix
            )
            .log_prob(torch.tensor([-1.0, -1.0]))
            .type(torch.float64),
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.distributions.MultivariateNormal(
                loc=means, covariance_matrix=covariance_matrix
            )
            .log_prob(torch.tensor([1.0, 1.0]))
            .type(torch.float64),
        ),
    ]


def _expected_independent_multivariate_normal_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    mean = torch.tensor([0.0])
    standard_deviation = torch.tensor([1.0])
    return [
        (
            torch.tensor([-1.0, -1.0, -1.0]),
            3
            * torch.distributions.Normal(loc=mean, scale=standard_deviation)
            .log_prob(torch.tensor([-1.0]))
            .type(torch.float64)[0],
        ),
        (
            torch.tensor([1.0, 1.0, 1.0]),
            3
            * torch.distributions.Normal(loc=mean, scale=standard_deviation)
            .log_prob(torch.tensor([1.0]))
            .type(torch.float64)[0],
        ),
        (
            torch.tensor([-1.0, 0.0, 1.0]),
            (
                torch.distributions.Normal(loc=mean, scale=standard_deviation).log_prob(
                    torch.tensor([-1.0])
                )[0]
                + torch.distributions.Normal(
                    loc=mean, scale=standard_deviation
                ).log_prob(torch.tensor([0.0]))[0]
                + torch.distributions.Normal(
                    loc=mean, scale=standard_deviation
                ).log_prob(torch.tensor([1.0]))[0]
            ).type(torch.float64),
        ),
    ]


def _expected_mixed_multivariate_independently_distribution() -> (
    list[tuple[Tensor, Tensor]]
):
    mean_normal_dist = 0.0
    standard_deviation_normal_dist = 1.0
    parameter_normal_dist = 0.0
    probability_normal_dist = torch.exp(
        torch.distributions.Normal(
            loc=mean_normal_dist, scale=standard_deviation_normal_dist
        ).log_prob(torch.tensor(parameter_normal_dist))
    ).type(torch.float64)

    return [
        (
            torch.tensor([0.0, parameter_normal_dist]),
            torch.log(0.5 * probability_normal_dist),
        ),
        (
            torch.tensor([-2.0, parameter_normal_dist]),
            torch.log(0.0 * probability_normal_dist),
        ),
        (
            torch.tensor([2.0, parameter_normal_dist]),
            torch.log(0.0 * probability_normal_dist),
        ),
    ]


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_uniform_distribution()
)
def test_univariate_uniform_distributed_prior(parameter: Tensor, expected: Tensor):
    lower_limit = -1.0
    upper_limit = 1.0
    sut = create_univariate_uniform_distribution(
        lower_limit=lower_limit, upper_limit=upper_limit, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_univariate_normal_distribution()
)
def test_univariate_normal_distributed_prior(parameter: Tensor, expected: Tensor):
    mean = 0.0
    standard_deviation = 1.0
    sut = create_univariate_normal_distribution(
        mean=mean, standard_deviation=standard_deviation, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_multivariate_normal_distribution()
)
def test_multivariate_normal_distributed_prior(parameter: Tensor, expected: Tensor):
    means = torch.tensor([0.0, 0.0])
    covariance_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    sut = create_multivariate_normal_distribution(
        means=means, covariance_matrix=covariance_matrix, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"), _expected_independent_multivariate_normal_distribution()
)
def test_independent_multivariate_normal_distributed_prior(
    parameter: Tensor, expected: Tensor
):
    means = torch.tensor([0.0, 0.0, 0.0])
    standard_deviations = torch.tensor([1.0, 1.0, 1.0])
    sut = create_independent_multivariate_normal_distribution(
        means=means, standard_deviations=standard_deviations, device=device
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "expected"),
    _expected_mixed_multivariate_independently_distribution(),
)
def test_mixed_multivariate_independently_distributed_prior(
    parameter: Tensor, expected: Tensor
):
    lower_bound_uniform_dist = -1
    upper_bound_uniform_dist = 1
    uniform_dist = create_univariate_uniform_distribution(
        lower_limit=lower_bound_uniform_dist,
        upper_limit=upper_bound_uniform_dist,
        device=device,
    )
    mean_normal_dist = 0.0
    standard_deviation_normal_dist = 1.0
    normal_dist = create_univariate_normal_distribution(
        mean=mean_normal_dist,
        standard_deviation=standard_deviation_normal_dist,
        device=device,
    )
    sut = create_mixed_multivariate_independently_distribution(
        independent_univariate_distributions=[uniform_dist, normal_dist]
    )

    actual = sut.log_prob(parameter)

    torch.testing.assert_close(actual, expected)
