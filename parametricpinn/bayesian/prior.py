from typing import TypeAlias, Union

import torch

from parametricpinn.statistics.distributions import (
    IndependentMultivariateNormalDistributon,
    MixedIndependetMultivariateDistribution,
    MultivariateNormalDistributon,
    UnivariateDistributions,
    UnivariateNormalDistributon,
    UnivariateUniformDistributon,
    create_independent_multivariate_normal_distribution,
    create_mixed_independent_multivariate_distribution,
    create_multivariate_normal_distribution,
    create_univariate_normal_distribution,
    create_univariate_uniform_distribution,
)
from parametricpinn.types import Device, Tensor

PriorDistribution: TypeAlias = Union[
    UnivariateUniformDistributon,
    UnivariateNormalDistributon,
    MultivariateNormalDistributon,
    MixedIndependetMultivariateDistribution,
    IndependentMultivariateNormalDistributon,
]


class Prior:
    def __init__(self, distribution: PriorDistribution):
        self.distribution = distribution
        self.dim = distribution.dim

    def prob(self, parameters: Tensor) -> Tensor:
        with torch.no_grad():
            return self._prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        with torch.no_grad():
            return self._log_prob(parameters)

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        with torch.autograd.set_detect_anomaly(True):
            return torch.autograd.grad(
                self._log_prob(parameters),
                parameters,
                retain_graph=False,
                create_graph=False,
            )[0]

    def sample(self) -> Tensor:
        return self.distribution.sample()

    def _prob(self, parameters: Tensor) -> Tensor:
        return torch.exp(self._log_prob(parameters))

    def _log_prob(self, parameters: Tensor) -> Tensor:
        return self.distribution.log_prob(parameters)


class MultipliedPriors(Prior):
    def __init__(self, priors: list[Prior]):
        self._priors = priors
        self._prior_dims = [prior.dim for prior in priors]
        self.dim = sum(self._prior_dims)

    def sample(self) -> Tensor:
        samples = [prior.distribution.sample() for prior in self._priors]
        return torch.concat(samples, dim=0)

    def _log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = []
        start_index = 0
        for i, prior in enumerate(self._priors[:-1]):
            dim_parameters_i = self._prior_dims[i]
            parameters_i = parameters[start_index : start_index + dim_parameters_i]
            log_probs.append(torch.unsqueeze(prior.log_prob(parameters_i), dim=0))
            start_index += dim_parameters_i
        parameters_last = parameters[start_index:]
        log_probs.append(
            torch.unsqueeze(self._priors[-1].log_prob(parameters_last), dim=0)
        )
        return torch.sum(torch.concat(log_probs), dim=0)


def create_univariate_uniform_distributed_prior(
    lower_limit: float, upper_limit: float, device: Device
) -> Prior:
    distribution = create_univariate_uniform_distribution(
        lower_limit, upper_limit, device
    )
    return Prior(distribution)


def create_univariate_normal_distributed_prior(
    mean: float, standard_deviation: float, device: Device
) -> Prior:
    distribution = create_univariate_normal_distribution(
        mean, standard_deviation, device
    )
    return Prior(distribution)


def create_multivariate_normal_distributed_prior(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> Prior:
    distribution = create_multivariate_normal_distribution(
        means, covariance_matrix, device
    )
    return Prior(distribution)


def create_independent_multivariate_normal_distributed_prior(
    means: Tensor, standard_deviations: Tensor, device: Device
) -> Prior:
    distribution = create_independent_multivariate_normal_distribution(
        means, standard_deviations, device
    )
    return Prior(distribution)


def create_mixed_independent_multivariate_distributed_prior(
    independent_univariate_distributions: list[UnivariateDistributions],
) -> Prior:
    distribution = create_mixed_independent_multivariate_distribution(
        independent_univariate_distributions
    )
    return Prior(distribution)


def multiply_priors(priors: list[Prior]) -> MultipliedPriors:
    return MultipliedPriors(priors)
