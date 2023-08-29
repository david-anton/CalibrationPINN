from typing import Protocol, TypeAlias, Union

import torch

from parametricpinn.errors import (
    MixedDistributionError,
    UnivariateNormalDistributionError,
    UnivariateUniformDistributionError,
)
from parametricpinn.types import Device, Tensor


def squeeze_if_necessary(log_prob: Tensor) -> Tensor:
    if log_prob.shape is not torch.Size([]):
        return torch.squeeze(log_prob, dim=0)
    return log_prob


class CustomDistribution(Protocol):
    def log_prob(self, sample: Tensor):
        pass


class UnivariateUniformDistributon:
    def __init__(self, lower_limit: float, upper_limit: float, device: Device):
        self._distribution = torch.distributions.Uniform(
            low=torch.tensor(lower_limit, dtype=torch.float64, device=device),
            high=torch.tensor(upper_limit, dtype=torch.float64, device=device),
            validate_args=False,
        )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise UnivariateUniformDistributionError(
                f"Unexpected shape of sample: {shape}."
            )

    def log_prob(self, sample: Tensor):
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)


class UnivariateNormalDistributon:
    def __init__(self, mean: float, standard_deviation: float, device: Device):
        self._distribution = torch.distributions.MultivariateNormal(
            loc=torch.tensor([mean], dtype=torch.float64, device=device),
            covariance_matrix=torch.tensor(
                [[standard_deviation**2]], dtype=torch.float64, device=device
            ),
        )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise UnivariateNormalDistributionError(
                f"Unexpected shape of sample: {shape}."
            )

    def log_prob(self, sample: Tensor):
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)


class MultivariateNormalDistributon:
    def __init__(self, means: Tensor, covariance_matrix: Tensor, device: Device):
        self._distribution = torch.distributions.MultivariateNormal(
            loc=means.type(torch.float64).to(device),
            covariance_matrix=covariance_matrix.type(torch.float64).to(device),
        )

    def log_prob(self, sample: Tensor):
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)


UnivariateDistributions: TypeAlias = Union[
    UnivariateUniformDistributon, UnivariateNormalDistributon
]


class MixedIndependetMultivariateDistribution:
    def __init__(self, distributions: list[UnivariateDistributions]) -> None:
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
        log_prob = torch.sum(
            torch.tensor(
                [
                    self._distributions[i].log_prob(torch.tensor([sample[i]]))
                    for i in range(sample.size()[0])
                ]
            )
        )
        return squeeze_if_necessary(log_prob)


def create_univariate_uniform_distribution(
    lower_limit: float, upper_limit: float, device: Device
) -> UnivariateUniformDistributon:
    return UnivariateUniformDistributon(lower_limit, upper_limit, device)


def create_univariate_normal_distribution(
    mean: float, standard_deviation: float, device: Device
) -> UnivariateNormalDistributon:
    return UnivariateNormalDistributon(mean, standard_deviation, device)


def create_multivariate_normal_distribution(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> MultivariateNormalDistributon:
    return MultivariateNormalDistributon(means, covariance_matrix, device)


def create_mixed_multivariate_independently_distribution(
    independent_univariate_distributions: list[UnivariateDistributions],
) -> MixedIndependetMultivariateDistribution:
    return MixedIndependetMultivariateDistribution(independent_univariate_distributions)
