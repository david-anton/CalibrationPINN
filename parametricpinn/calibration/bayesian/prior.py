from typing import Callable, TypeAlias, Union

import torch

from parametricpinn.calibration.bayesian.distributions import (
    MixedIndependetMultivariateDistribution,
    TorchUnivariateDistributions,
)
from parametricpinn.types import (
    Device,
    Tensor,
    TorchMultiNormalDist,
    TorchUniformDist,
    TorchUniNormalDist,
)

PriorDistribution: TypeAlias = Union[
    TorchUniformDist,
    TorchUniNormalDist,
    TorchMultiNormalDist,
    MixedIndependetMultivariateDistribution,
]
PriorFunc: TypeAlias = Callable[[Tensor], Tensor]


def compile_prior(distribution: PriorDistribution) -> PriorFunc:
    def prior_func(parameters: Tensor) -> Tensor:
        return torch.exp(distribution.log_prob(parameters))

    return prior_func


def compile_univariate_uniform_distributed_prior(
    lower_limit: float, upper_limit: float, device: Device
) -> PriorFunc:
    distribution = torch.distributions.Uniform(
        low=torch.tensor(lower_limit, device=device),
        high=torch.tensor(upper_limit, device=device),
        validate_args=False,
    )
    return compile_prior(distribution)


def compile_univariate_normal_distributed_prior(
    mean: float, standard_deviation: float, device: Device
) -> PriorFunc:
    distribution = torch.distributions.Normal(
        loc=torch.tensor(mean, device=device),
        scale=torch.tensor(standard_deviation, device=device),
    )
    return compile_prior(distribution)


def compile_multivariate_normal_distributed_prior(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> PriorFunc:
    distribution = torch.distributions.MultivariateNormal(
        loc=means.to(device),
        covariance_matrix=covariance_matrix.to(device),
    )
    return compile_prior(distribution)


def compile_mixed_multivariate_independently_distributed_prior(
    independent_univariate_distributions: list[TorchUnivariateDistributions],
) -> PriorFunc:
    distribution = MixedIndependetMultivariateDistribution(
        independent_univariate_distributions
    )
    return compile_prior(distribution)
