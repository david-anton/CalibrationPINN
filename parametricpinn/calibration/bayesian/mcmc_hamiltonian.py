from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.calibration.bayesian.mcmc_base import (
    compile_unnnormalized_posterior,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.plot import plot_posterior_normal_distributions
from parametricpinn.calibration.bayesian.statistics import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, NPArray, Tensor, TorchMultiNormalDist

Samples: TypeAlias = list[Tensor]
MCMC_Hamiltonian_func: TypeAlias = Callable[
    [
        tuple[str, ...],
        tuple[float, ...],
        LikelihoodFunc,
        TorchMultiNormalDist,
        Tensor,
        int,
        float,
        int,
        str,
        ProjectDirectory,
        Device,
    ],
    tuple[MomentsMultivariateNormal, NPArray],
]
PotentialEnergyFunc: TypeAlias = Callable[[Tensor], Tensor]
KineticEnergyFunc: TypeAlias = Callable[[Tensor], Tensor]


@dataclass
class HamiltonianConfig(MCMCConfig):
    num_leabfrog_steps: int
    leapfrog_step_size: float


def mcmc_hamiltonian(
    parameter_names: tuple[str, ...],
    true_parameters: tuple[float, ...],
    likelihood: LikelihoodFunc,
    prior: TorchMultiNormalDist,
    initial_parameters: Tensor,
    num_leapfrog_steps: int,
    leapfrog_step_size: float,
    num_iterations: int,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    unnormalized_likelihood = compile_unnnormalized_posterior(
        likelihood=likelihood, prior=prior
    )

    def _compile_potential_energy_func() -> PotentialEnergyFunc:
        def _potential_energy_func(parameters: Tensor) -> Tensor:
            return -torch.log(unnormalized_likelihood(parameters))

        return _potential_energy_func

    def _compile_kinetic_energy_func() -> KineticEnergyFunc:
        def _kinetic_energy_func(momentums: Tensor) -> Tensor:
            return 1 / 2 * torch.sum(torch.pow(momentums, 2))

        return _kinetic_energy_func

    def _draw_momentums(parameters: Tensor) -> Tensor:
        mean = torch.zeros_like(parameters, device=device)
        covariance = torch.diag(torch.ones_like(parameters, device=device))
        return torch.distributions.multivariate_normal.MultivariateNormal(
            loc=mean, covariance_matrix=covariance
        ).sample(sample_shape=parameters.size())

    def _propose_next_parameters(
        parameters: Tensor, momentums: Tensor
    ) -> tuple[Tensor, Tensor]:
        potential_energy_func = _compile_potential_energy_func()
        step_size = torch.tensor(leapfrog_step_size)

        def _leapfrog_step(
            parameters: Tensor, momentums: Tensor, is_last_step: bool
        ) -> tuple[Tensor, Tensor]:
            parameters = parameters + step_size * momentums
            if not is_last_step:
                momentums = momentums - step_size * torch.autograd.grad(
                    potential_energy_func(parameters), parameters
                )
            return parameters, momentums

        # Half step for momentums (in the beginning)
        momentums = momentums - 1 / 2 * step_size * torch.autograd.grad(
            potential_energy_func(parameters), parameters
        )
        for i in range(num_leapfrog_steps):
            is_last_step = i == num_leapfrog_steps - 1
            parameters, momentums = _leapfrog_step(parameters, momentums, is_last_step)

        # Half step for momentums (in the end)
        momentums = momentums - 1 / 2 * step_size * torch.autograd.grad(
            potential_energy_func(parameters), parameters
        )
        return parameters, momentums

    def _one_iteration(parameters: Tensor, momentums: Tensor) -> Tensor:
        potential_energy_func = _compile_potential_energy_func()
        kinetic_energy_func = _compile_kinetic_energy_func()
        potential_energy = potential_energy_func(parameters)
        kinetic_energy = kinetic_energy_func(momentums)
        next_parameters, next_momentums = _propose_next_parameters(
            parameters, momentums
        )
        # Negate momentums to make proposal symmetric
        next_momentums = -next_momentums
        next_potential_energy = potential_energy_func(next_parameters)
        next_kinetic_energy = kinetic_energy_func(next_momentums)

        acceptance_ratio = torch.minimum(
            torch.tensor(1.0, device=device),
            torch.exp(
                potential_energy
                - next_potential_energy
                + kinetic_energy
                - next_kinetic_energy
            ),
        )
        rand_uniform_number = torch.squeeze(torch.rand(1, device=device), 0)
        if rand_uniform_number > acceptance_ratio:
            next_parameters = parameters
        return next_parameters

    samples_list: Samples = []
    parameters = initial_parameters
    for _ in range(num_iterations):
        momentums = _draw_momentums(parameters)
        parameters = _one_iteration(parameters, momentums)
        samples_list.append(parameters)

    samples = torch.stack(samples_list, dim=0).detach().cpu().numpy()
    posterior_moments = determine_moments_of_multivariate_normal_distribution(samples)
    plot_posterior_normal_distributions(
        parameter_names,
        true_parameters,
        posterior_moments,
        samples,
        output_subdir,
        project_directory,
    )

    return posterior_moments, samples
