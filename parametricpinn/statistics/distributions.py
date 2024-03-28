from typing import Protocol, TypeAlias, Union

import torch

from parametricpinn.errors import (
    IndependentMultivariateNormalDistributionError,
    MixedDistributionError,
    MultivariateNormalDistributionError,
    MultivariateUniformDistributionError,
    UnivariateNormalDistributionError,
    UnivariateUniformDistributionError,
    GammaDistributionError,
)
from parametricpinn.types import Device, Tensor


def squeeze_if_necessary(log_prob: Tensor) -> Tensor:
    if log_prob.shape is not torch.Size([]):
        return torch.squeeze(log_prob, dim=0)
    return log_prob


class CustomDistribution(Protocol):
    def log_prob(self, sample: Tensor) -> Tensor:
        pass

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        pass


class UnivariateUniformDistributon:
    def __init__(self, lower_limit: float, upper_limit: float, device: Device):
        self._distribution = torch.distributions.Uniform(
            low=torch.tensor(lower_limit, dtype=torch.float64, device=device),
            high=torch.tensor(upper_limit, dtype=torch.float64, device=device),
            validate_args=False,
        )
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise UnivariateUniformDistributionError(
                f"Unexpected shape of sample: {shape}."
            )


class MultivariateUniformDistribution:
    def __init__(self, lower_limits: Tensor, upper_limits: Tensor, device: Device):
        self._validate_limits(lower_limits, upper_limits)
        self._distribution = torch.distributions.Uniform(
            low=lower_limits.type(torch.float64).to(device),
            high=upper_limits.type(torch.float64).to(device),
            validate_args=False,
        )
        self.dim = torch.numel(lower_limits)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = torch.sum(self._distribution.log_prob(sample), dim=0)
        return squeeze_if_necessary(log_prob)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    def _validate_limits(self, lower_limits: Tensor, upper_limits: Tensor) -> None:
        shape_lower_limits = lower_limits.size()
        shape_upper_limits = upper_limits.size()
        if not shape_lower_limits == shape_upper_limits:
            raise MultivariateUniformDistributionError(
                "Different shape of lower and upper limits: {shape_lower_limits} and {shape_upper_limits}"
            )

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise MultivariateUniformDistributionError(
                f"Unexpected shape of sample: {shape}."
            )


class UnivariateNormalDistributon:
    def __init__(self, mean: float, standard_deviation: float, device: Device):
        self._distribution = torch.distributions.MultivariateNormal(
            loc=torch.tensor([mean], dtype=torch.float64, device=device),
            covariance_matrix=torch.tensor(
                [[standard_deviation**2]], dtype=torch.float64, device=device
            ),
            validate_args=False,
        )
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise UnivariateNormalDistributionError(
                f"Unexpected shape of sample: {shape}."
            )


class MultivariateNormalDistributon:
    def __init__(self, means: Tensor, covariance_matrix: Tensor, device: Device):
        self._distribution = torch.distributions.MultivariateNormal(
            loc=means.type(torch.float64).to(device),
            covariance_matrix=covariance_matrix.type(torch.float64).to(device),
            validate_args=False,
        )
        self.means = self._distribution.mean
        self.variances = self._distribution.variance
        self.dim = torch.numel(means)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise MultivariateNormalDistributionError(
                f"Unexpected shape of sample: {shape}."
            )


class IndependentMultivariateNormalDistributon:
    def __init__(self, means: Tensor, standard_deviations: Tensor, device: Device):
        self._distribution = torch.distributions.Normal(
            loc=means.type(torch.float64).to(device),
            scale=standard_deviations.type(torch.float64).to(device),
            validate_args=False,
        )
        self.means = self._distribution.mean
        self.standard_deviations = self._distribution.stddev
        self.dim = torch.numel(means)

    def log_probs_individual(self, sample: Tensor) -> Tensor:
        return self._distribution.log_prob(sample)

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        log_probs_individual = self.log_probs_individual(sample)
        log_prob = torch.sum(log_probs_individual)
        return squeeze_if_necessary(log_prob)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not shape == torch.Size([self.dim]):
            raise IndependentMultivariateNormalDistributionError(
                f"Unexpected shape of sample: {shape}."
            )


class GammaDistribution:
    def __init__(self, concentration: float, rate: float, device: Device) -> None:
        self._distribution = torch.distributions.Gamma(
            concentration=torch.tensor(
                concentration, dtype=torch.float64, device=device
            ),
            rate=torch.tensor(rate, dtype=torch.float64, device=device),
            validate_args=False,
        )
        self._device = device
        self.dim = 1

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        if torch.all(sample < 0.0):
            log_prob = torch.tensor(-torch.inf, device=self._device)
        else:
            log_prob = self._distribution.log_prob(sample)
        return squeeze_if_necessary(log_prob)

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        return self._distribution.rsample(sample_shape)

    def _validate_sample(self, sample: Tensor) -> None:
        shape = sample.shape
        if not (shape == torch.Size([1]) or shape == torch.Size([])):
            raise GammaDistributionError(f"Unexpected shape of sample: {shape}.")


UnivariateDistributions: TypeAlias = Union[
    UnivariateUniformDistributon, UnivariateNormalDistributon
]


class MixedIndependetMultivariateDistribution:
    def __init__(self, distributions: list[UnivariateDistributions]) -> None:
        self._distributions = distributions
        self.dim = len(distributions)

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

    def sample(self, sample_shape=torch.Size()) -> Tensor:
        if sample_shape == torch.Size():
            samples = [
                distribution.sample(sample_shape)
                for distribution in self._distributions
            ]
            return torch.concat(samples, dim=0)
        else:
            samples = [
                distribution.sample(sample_shape)
                for distribution in self._distributions
            ]
            return torch.concat(samples, dim=1)

    def _validate_sample(self, sample: Tensor) -> None:
        if sample.dim() != 1:
            raise MixedDistributionError(
                "Sample is expected to be a one-dimensional tensor."
            )
        if sample.size()[0] != len(self._distributions):
            raise MixedDistributionError(
                "The size of sample does not match the number of mixed distributions."
            )


def create_univariate_uniform_distribution(
    lower_limit: float, upper_limit: float, device: Device
) -> UnivariateUniformDistributon:
    return UnivariateUniformDistributon(lower_limit, upper_limit, device)


def create_multivariate_uniform_distribution(
    lower_limits: Tensor, upper_limits: Tensor, device: Device
) -> MultivariateUniformDistribution:
    return MultivariateUniformDistribution(lower_limits, upper_limits, device)


def create_univariate_normal_distribution(
    mean: float, standard_deviation: float, device: Device
) -> UnivariateNormalDistributon:
    return UnivariateNormalDistributon(mean, standard_deviation, device)


def create_multivariate_normal_distribution(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> MultivariateNormalDistributon:
    return MultivariateNormalDistributon(means, covariance_matrix, device)


def create_independent_multivariate_normal_distribution(
    means: Tensor, standard_deviations: Tensor, device: Device
) -> IndependentMultivariateNormalDistributon:
    return IndependentMultivariateNormalDistributon(means, standard_deviations, device)


def create_gamma_distribution(
    concentration: float, rate: float, device: Device
) -> GammaDistribution:
    return GammaDistribution(concentration, rate, device)


def create_mixed_independent_multivariate_distribution(
    independent_univariate_distributions: list[UnivariateDistributions],
) -> MixedIndependetMultivariateDistribution:
    return MixedIndependetMultivariateDistribution(independent_univariate_distributions)
