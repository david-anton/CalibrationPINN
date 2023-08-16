from typing import Callable, TypeAlias, Union

import torch

from parametricpinn.calibration.bayesian.mcmc_base import (
    Parameters,
    UnnormalizedPosterior,
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


PotentialEnergyFunc: TypeAlias = Callable[[Parameters], Energy]


def _potential_energy_func(
    unnormalized_posterior: UnnormalizedPosterior,
) -> PotentialEnergyFunc:
    def potential_energy_func(parameters: Parameters) -> Energy:
        return -torch.log(unnormalized_posterior(parameters))

    return potential_energy_func


def kinetic_energy_func(momentums: Momentums) -> Energy:
    return 1 / 2 * torch.sum(torch.pow(momentums, 2))


DrawMomentumsFunc: TypeAlias = Callable[[], Momentums]
MomentumsDistribution = Union[TorchUniNormalDist, TorchMultiNormalDist]


def _sample_normalized_momentums(
    parameters: Parameters,
    device: Device,
) -> DrawMomentumsFunc:
    def compile_momentum_distribution() -> MomentumsDistribution:
        if parameters.size() == (1,):
            mean = torch.tensor([0.0], dtype=torch.float64, device=device)
            standard_deviation = torch.tensor([1.0], dtype=torch.float64, device=device)
            return torch.distributions.Normal(loc=mean, scale=standard_deviation)
        else:
            means = torch.zeros_like(parameters, dtype=torch.float64, device=device)
            covariance_matrix = torch.diag(
                torch.ones_like(parameters, dtype=torch.float64, device=device)
            )
            return torch.distributions.MultivariateNormal(
                loc=means, covariance_matrix=covariance_matrix
            )

    momentums_dist = compile_momentum_distribution()

    def sample_momentums() -> Momentums:
        momentums = momentums_dist.sample()
        return momentums.requires_grad_(True)

    return sample_momentums
