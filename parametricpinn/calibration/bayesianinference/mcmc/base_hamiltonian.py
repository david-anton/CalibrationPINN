from typing import Callable, TypeAlias, Union

import torch

from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.bayesianinference.mcmc.base import (
    LogUnnormalizedPosterior,
    Parameters,
    Probability,
)
from parametricpinn.types import (
    Device,
    Tensor,
    TorchMultiNormalDist,
    TorchUniNormalDist,
)

Momentums: TypeAlias = Tensor
Energy: TypeAlias = Tensor
StepSizes: TypeAlias = Tensor


GradLogUnnormalizedPosterior: TypeAlias = Callable[[Tensor], Tensor]


def _grad_log_unnormalized_posterior(
    likelihood: Likelihood, prior: Prior
) -> GradLogUnnormalizedPosterior:
    def grad_log_unnormalized_posterior(parameters: Parameters) -> Probability:
        return likelihood.grad_log_prob(parameters) + prior.grad_log_prob(parameters)

    return grad_log_unnormalized_posterior


PotentialEnergyFunc: TypeAlias = Callable[[Parameters], Energy]


def _potential_energy_func(
    log_unnormalized_posterior: LogUnnormalizedPosterior,
) -> PotentialEnergyFunc:
    def potential_energy_func(parameters: Parameters) -> Energy:
        return -log_unnormalized_posterior(parameters)

    return potential_energy_func


GradPotentialEnergyFunc: TypeAlias = Callable[[Parameters], Energy]


def _grad_potential_energy_func(
    grad_log_unnormalized_posterior: GradLogUnnormalizedPosterior,
) -> GradPotentialEnergyFunc:
    def grad_potential_energy_func(parameters: Parameters) -> Energy:
        return -grad_log_unnormalized_posterior(parameters)

    return grad_potential_energy_func


def kinetic_energy_func(momentums: Momentums) -> Energy:
    return 1 / 2 * torch.sum(torch.pow(momentums, 2))


DrawMomentumsFunc: TypeAlias = Callable[[], Momentums]
MomentumsDistribution = Union[TorchUniNormalDist, TorchMultiNormalDist]


def _sample_normalized_momentums(
    parameters: Parameters,
    device: Device,
) -> DrawMomentumsFunc:
    def compile_momentum_distribution() -> MomentumsDistribution:
        if parameters.size() == torch.Size([]):
            mean = torch.tensor([0.0], dtype=torch.float64, device=device)
            standard_deviation = torch.tensor([1.0], dtype=torch.float64, device=device)
            return torch.distributions.Normal(loc=mean, scale=standard_deviation)
        else:
            means = torch.zeros_like(parameters, dtype=torch.float64, device=device)
            standard_deviations = torch.ones_like(
                parameters, dtype=torch.float64, device=device
            )
            return torch.distributions.Normal(loc=means, scale=standard_deviations)

    momentums_dist = compile_momentum_distribution()

    def sample_momentums() -> Momentums:
        momentums = momentums_dist.sample()
        return momentums.requires_grad_(True)

    return sample_momentums
