from typing import TypeAlias, Union

import torch

from parametricpinn.errors import MixedDistributionError
from parametricpinn.types import (
    Device,
    Tensor,
    TorchMultiNormalDist,
    TorchUniformDist,
    TorchUniNormalDist,
)

# There is no explicit univariate uniform distribution in Pytorch.
TorchUnivariateDistributions: TypeAlias = Union[TorchUniformDist, TorchUniNormalDist]


class MixedIndependetMultivariateDistribution:
    def __init__(self, distributions: list[TorchUnivariateDistributions]) -> None:
        self._distributions = distributions

    def _validate_value(self, value: Tensor) -> None:
        if value.dim() != 1:
            raise MixedDistributionError(
                "Value is expected to be a one-dimensional tensor."
            )
        if value.size()[0] != len(self._distributions):
            raise MixedDistributionError(
                "The size of value does not match the number of mixed distributions."
            )

    def log_prob(self, value: Tensor) -> Tensor:
        self._validate_value(value)
        return torch.sum(
            torch.tensor(
                [
                    self._distributions[i].log_prob(value[i])
                    for i in range(value.size()[0])
                ]
            )
        )


PriorDistribution: TypeAlias = Union[
    TorchUniformDist,
    TorchUniNormalDist,
    TorchMultiNormalDist,
    MixedIndependetMultivariateDistribution,
]


class Prior:
    def __init__(self, distribution: PriorDistribution):
        self._distribution = distribution

    def probability(self, parameters: Tensor) -> Tensor:
        return torch.pow(10, self._distribution.log_prob(parameters))


def compile_uniform_distributed_prior(
    lower_limit: Tensor, upper_limit: Tensor
) -> Prior:
    distribution = torch.distributions.Uniform(low=lower_limit, high=upper_limit)
    return Prior(distribution)


def compile_univariate_normal_distributed_prior(
    mean: float, standard_deviation: float, device: Device
) -> Prior:
    distribution = torch.distributions.Normal(
        loc=torch.tensor(mean, dtype=torch.float64, device=device),
        scale=torch.tensor(standard_deviation, dtype=torch.float64, device=device),
    )
    return Prior(distribution)


def compile_multivariate_normal_distributed_prior(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> Prior:
    distribution = torch.distributions.MultivariateNormal(
        loc=means.type(torch.float64).to(device),
        covariance_matrix=covariance_matrix.type(torch.float64).to(device),
    )
    return Prior(distribution)


def compile_mixed_multivariate_independently_distributed_prior(
    independent_univariate_distributions: list[TorchUnivariateDistributions],
) -> Prior:
    distribution = MixedIndependetMultivariateDistribution(
        independent_univariate_distributions
    )
    return Prior(distribution)
