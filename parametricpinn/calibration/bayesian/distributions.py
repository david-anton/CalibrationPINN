from typing import Protocol, TypeAlias, Union

import torch

from parametricpinn.errors import (
    MixedDistributionError,
    UnivariateNormalDistributionError,
)
from parametricpinn.types import Device, Tensor, TorchMultiNormalDist, TorchUniformDist


class CustomDistribution(Protocol):
    def log_prob(self, sample: Tensor):
        pass


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
        return self._distribution.log_prob(sample)


# There is no explicit univariate uniform distribution in Pytorch.
UnivariateDistributions: TypeAlias = Union[
    TorchUniformDist, UnivariateNormalDistributon
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
        return torch.sum(
            torch.tensor(
                [
                    self._distributions[i].log_prob(torch.tensor([sample[i]]))
                    for i in range(sample.size()[0])
                ]
            )
        )


def create_univariate_uniform_distribution(
    lower_limit: float, upper_limit: float, device: Device
) -> TorchUniformDist:
    return torch.distributions.Uniform(
        low=torch.tensor(lower_limit, dtype=torch.float64, device=device),
        high=torch.tensor(upper_limit, dtype=torch.float64, device=device),
        validate_args=False,
    )


def create_univariate_normal_distribution(
    mean: float, standard_deviation: float, device: Device
) -> UnivariateNormalDistributon:
    return UnivariateNormalDistributon(mean, standard_deviation, device)


def create_multivariate_normal_distribution(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> TorchMultiNormalDist:
    return torch.distributions.MultivariateNormal(
        loc=means.type(torch.float64).to(device),
        covariance_matrix=covariance_matrix.type(torch.float64).to(device),
    )


def create_mixed_multivariate_independently_distribution(
    independent_univariate_distributions: list[UnivariateDistributions],
) -> MixedIndependetMultivariateDistribution:
    return MixedIndependetMultivariateDistribution(independent_univariate_distributions)
