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

    def _validate_sample(self, sample: Tensor) -> None:
        if sample.dim() != 1:
            raise MixedDistributionError(
                "Sample is expected to be a one-dimensional tensor."
            )
        if sample.size()[0] != len(self._distributions):
            raise MixedDistributionError(
                "The size of sample does not match the number of mixed distributions."
            )

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        return torch.sum(
            torch.tensor(
                [
                    self._distributions[i].log_prob(sample[i])
                    for i in range(sample.size()[0])
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

    def __call__(self, parameters: Tensor) -> Tensor:
        return torch.exp(self._distribution.log_prob(parameters))


def compile_univariate_uniform_distributed_prior(
    lower_limit: float, upper_limit: float, device: Device
) -> Prior:
    distribution = torch.distributions.Uniform(
        low=torch.tensor(lower_limit, device=device),
        high=torch.tensor(upper_limit, device=device),
        validate_args=False,
    )
    return Prior(distribution)


def compile_univariate_normal_distributed_prior(
    mean: float, standard_deviation: float, device: Device
) -> Prior:
    distribution = torch.distributions.Normal(
        loc=torch.tensor(mean, device=device),
        scale=torch.tensor(standard_deviation, device=device),
    )
    return Prior(distribution)


def compile_multivariate_normal_distributed_prior(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> Prior:
    distribution = torch.distributions.MultivariateNormal(
        loc=means.to(device),
        covariance_matrix=covariance_matrix.to(device),
    )
    return Prior(distribution)


def compile_mixed_multivariate_independently_distributed_prior(
    independent_univariate_distributions: list[TorchUnivariateDistributions],
) -> Prior:
    distribution = MixedIndependetMultivariateDistribution(
        independent_univariate_distributions
    )
    return Prior(distribution)
