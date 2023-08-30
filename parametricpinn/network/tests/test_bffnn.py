from typing import cast

import torch

from parametricpinn.calibration.bayesian.distributions import (
    MultivariateNormalDistributon,
)
from parametricpinn.network.bffnn import BFFNN, ParameterPriorStds

device = torch.device("cpu")
layer_sizes = [2, 4, 1]
std_weight = 2.0
std_bias = 1.0


def test_create_multivariate_normal_prior_distribution() -> None:
    sut = BFFNN(layer_sizes=layer_sizes)
    parameter_stds = ParameterPriorStds(weight=std_weight, bias=std_bias)

    prior = sut.create_multivariate_normal_prior(
        parameter_stds=parameter_stds, device=device
    )
    actual = type(prior.distribution)

    expected = MultivariateNormalDistributon
    assert actual == expected


def test_create_multivariate_normal_prior_means() -> None:
    sut = BFFNN(layer_sizes=layer_sizes)
    parameter_stds = ParameterPriorStds(weight=std_weight, bias=std_bias)

    prior = sut.create_multivariate_normal_prior(
        parameter_stds=parameter_stds, device=device
    )
    distribution = cast(MultivariateNormalDistributon, prior.distribution)
    actual = distribution.means

    expected = torch.concat(
        [
            torch.zeros((8,)),
            torch.zeros((4,)),
            torch.zeros((4,)),
            torch.zeros((1,)),
        ],
        dim=0,
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_create_multivariate_normal_prior_variances() -> None:
    sut = BFFNN(layer_sizes=layer_sizes)
    parameter_stds = ParameterPriorStds(weight=std_weight, bias=std_bias)

    prior = sut.create_multivariate_normal_prior(
        parameter_stds=parameter_stds, device=device
    )
    distribution = cast(MultivariateNormalDistributon, prior.distribution)
    actual = distribution.variances

    expected = torch.concat(
        [
            torch.full((8,), std_weight**2),
            torch.full((4,), std_bias**2),
            torch.full((4,), std_weight**2),
            torch.full((1,), std_bias**2),
        ],
        dim=0,
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)
