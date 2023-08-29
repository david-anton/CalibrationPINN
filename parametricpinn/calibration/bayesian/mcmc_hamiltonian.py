from dataclasses import dataclass
from typing import Callable, TypeAlias, Union

import torch

from parametricpinn.calibration.bayesian.likelihood import Likelihood
from parametricpinn.calibration.bayesian.mcmc_base import (
    IsAccepted,
    Samples,
    _unnnormalized_posterior,
    evaluate_acceptance_ratio,
    expand_num_iterations,
    postprocess_samples,
    remove_burn_in_phase,
)
from parametricpinn.calibration.bayesian.mcmc_base_hamiltonian import (
    Momentums,
    Parameters,
    StepSizes,
    _potential_energy_func,
    _sample_normalized_momentums,
    kinetic_energy_func,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.prior import Prior
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, NPArray, Tensor

IsLastStep: TypeAlias = bool

MCMCHamiltonianFunc: TypeAlias = Callable[
    [
        tuple[str, ...],
        tuple[float, ...],
        Likelihood,
        Prior,
        Parameters,
        int,
        StepSizes,
        int,
        int,
        str,
        ProjectDirectory,
        Device,
    ],
    tuple[MomentsMultivariateNormal, NPArray],
]


@dataclass
class HamiltonianConfig(MCMCConfig):
    num_leabfrog_steps: int
    leapfrog_step_sizes: Tensor


def mcmc_hamiltonian(
    parameter_names: tuple[str, ...],
    true_parameters: tuple[float, ...],
    likelihood: Likelihood,
    prior: Prior,
    initial_parameters: Parameters,
    num_leapfrog_steps: int,
    leapfrog_step_sizes: StepSizes,
    num_iterations: int,
    num_burn_in_iterations: int,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    step_sizes = leapfrog_step_sizes
    num_total_iterations = expand_num_iterations(num_iterations, num_burn_in_iterations)

    unnormalized_posterior = _unnnormalized_posterior(likelihood, prior)
    draw_normalized_momentums_func = _sample_normalized_momentums(
        initial_parameters, device
    )
    potential_energy_func = _potential_energy_func(unnormalized_posterior)

    def hamiltonian_sampler(parameters: Parameters) -> tuple[Parameters, IsAccepted]:
        def propose_next_state(
            parameters: Tensor, momentums: Tensor
        ) -> tuple[Tensor, Tensor]:
            def half_momentums_step(
                parameters: Parameters, momentums: Momentums
            ) -> Momentums:
                return (
                    momentums
                    - step_sizes
                    / 2
                    * torch.autograd.grad(
                        potential_energy_func(parameters),
                        parameters,
                        retain_graph=False,
                        create_graph=False,
                    )[0]
                )

            def full_parameter_and_momentum_step(
                parameters: Parameters, momentums: Momentums, is_last_step: IsLastStep
            ) -> tuple[Tensor, Tensor]:
                parameters = parameters + step_sizes * momentums
                if not is_last_step:
                    momentums = (
                        momentums
                        - step_sizes
                        * torch.autograd.grad(
                            potential_energy_func(parameters),
                            parameters,
                            retain_graph=False,
                            create_graph=False,
                        )[0]
                    )
                return parameters, momentums

            # Half step for momentum (in the beginning)
            momentums = half_momentums_step(parameters, momentums)

            # Full steps
            for i in range(num_leapfrog_steps):
                is_last_step = i == num_leapfrog_steps - 1
                parameters, momentums = full_parameter_and_momentum_step(
                    parameters, momentums, is_last_step
                )

            # Half step for momentums (in the end)
            momentums = half_momentums_step(parameters, momentums)
            return parameters, momentums

        @dataclass
        class MHUpdateState:
            parameters: Tensor
            momentums: Tensor
            next_parameters: Tensor
            next_momentums: Tensor

        def metropolis_hastings_update(
            state: MHUpdateState,
        ) -> tuple[Parameters, IsAccepted]:
            potential_energy = potential_energy_func(state.parameters)
            kinetic_energy = kinetic_energy_func(state.momentums)
            # Negate momentums to make proposal symmetric
            state.next_momentums = -state.next_momentums
            with torch.no_grad():
                next_potential_energy = potential_energy_func(state.next_parameters)
                next_kinetic_energy = kinetic_energy_func(state.next_momentums)

            acceptance_prob = torch.exp(
                potential_energy
                - next_potential_energy
                + kinetic_energy
                - next_kinetic_energy
            )
            rand_uniform_number = torch.squeeze(torch.rand(1, device=device), 0)
            next_parameters = state.next_parameters
            is_accepted = True
            if rand_uniform_number > acceptance_prob:
                next_parameters = state.parameters
                is_accepted = False
            return next_parameters, is_accepted

        momentums = draw_normalized_momentums_func()
        next_parameters, next_momentums = propose_next_state(parameters, momentums)
        mh_update_state = MHUpdateState(
            parameters=parameters,
            momentums=momentums,
            next_parameters=next_parameters,
            next_momentums=next_momentums,
        )
        return metropolis_hastings_update(mh_update_state)

    def one_iteration(parameters: Tensor) -> tuple[Parameters, IsAccepted]:
        return hamiltonian_sampler(parameters)

    samples_list: Samples = []
    num_accepted_proposals = 0
    parameters = initial_parameters
    for i in range(num_total_iterations):
        parameters = parameters.clone().type(torch.float64).requires_grad_(True)
        parameters, is_accepted = one_iteration(parameters)
        parameters.detach()
        samples_list.append(parameters)
        if i > num_burn_in_iterations and is_accepted:
            num_accepted_proposals += 1

    samples_list = remove_burn_in_phase(
        sample_list=samples_list, num_burn_in_iterations=num_burn_in_iterations
    )
    moments, samples = postprocess_samples(
        samples_list=samples_list,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        mcmc_algorithm="hamiltonian_mcmc",
        output_subdir=output_subdir,
        project_directory=project_directory,
    )
    evaluate_acceptance_ratio(
        num_accepted_proposals=num_accepted_proposals, num_iterations=num_iterations
    )

    return moments, samples
