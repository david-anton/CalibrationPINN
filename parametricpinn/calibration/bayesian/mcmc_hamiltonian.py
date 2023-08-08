from dataclasses import dataclass
from typing import Callable, TypeAlias, Union

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.calibration.bayesian.mcmc_base import (
    Samples,
    compile_unnnormalized_posterior,
    correct_num_iterations,
    evaluate_acceptance_ratio,
    postprocess_samples,
    remove_burn_in_phase,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import (
    Device,
    NPArray,
    Tensor,
    TorchMultiNormalDist,
    TorchUniNormalDist,
)

MCMCHamiltonianFunc: TypeAlias = Callable[
    [
        tuple[str, ...],
        tuple[float, ...],
        LikelihoodFunc,
        TorchMultiNormalDist,
        Tensor,
        int,
        Tensor,
        int,
        int,
        str,
        ProjectDirectory,
        Device,
    ],
    tuple[MomentsMultivariateNormal, NPArray],
]
PotentialEnergyFunc: TypeAlias = Callable[[Tensor], Tensor]
KineticEnergyFunc: TypeAlias = Callable[[Tensor], Tensor]
MomentumsDistribution = Union[TorchUniNormalDist, TorchMultiNormalDist]
DrawMomentumsFunc: TypeAlias = Callable[[], Tensor]


@dataclass
class HamiltonianConfig(MCMCConfig):
    num_leabfrog_steps: int
    leapfrog_step_sizes: Tensor


def mcmc_hamiltonian(
    parameter_names: tuple[str, ...],
    true_parameters: tuple[float, ...],
    likelihood: LikelihoodFunc,
    prior: TorchMultiNormalDist,
    initial_parameters: Tensor,
    num_leapfrog_steps: int,
    leapfrog_step_sizes: Tensor,
    num_iterations: int,
    num_burn_in_iterations: int,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    print("MCMC algorithm used: Hamiltonian")

    num_total_iterations = correct_num_iterations(
        num_iterations=num_iterations, num_burn_in_iterations=num_burn_in_iterations
    )
    unnormalized_posterior = compile_unnnormalized_posterior(
        likelihood=likelihood, prior=prior
    )

    def compile_potential_energy_func() -> PotentialEnergyFunc:
        def potential_energy_func(parameters: Tensor) -> Tensor:
            return -torch.log(unnormalized_posterior(parameters))

        return potential_energy_func

    def compile_kinetic_energy_func() -> KineticEnergyFunc:
        def kinetic_energy_func(momentums: Tensor) -> Tensor:
            # Simplest form of kinetic energy
            return 1 / 2 * torch.sum(torch.pow(momentums, 2))

        return kinetic_energy_func

    def compile_draw_normalized_momentums_func(parameters: Tensor) -> DrawMomentumsFunc:
        def compile_momentum_distribution() -> MomentumsDistribution:
            if parameters.size() == (1,):
                mean = torch.tensor(0.0, device=device)
                standard_deviation = torch.tensor(1.0, device=device)
                return torch.distributions.Normal(loc=mean, scale=standard_deviation)
            else:
                means = torch.zeros_like(parameters, device=device)
                covariance_matrix = torch.diag(
                    torch.ones_like(parameters, device=device)
                )
                return torch.distributions.MultivariateNormal(
                    loc=means, covariance_matrix=covariance_matrix
                )

        momentums_dist = compile_momentum_distribution()

        def draw_momentums_func() -> Tensor:
            momentums = momentums_dist.sample()
            return momentums.requires_grad_(True)

        return draw_momentums_func

    def propose_next_parameters(
        parameters: Tensor, momentums: Tensor
    ) -> tuple[Tensor, Tensor]:
        potential_energy_func = compile_potential_energy_func()
        step_sizes = leapfrog_step_sizes

        def leapfrog_step(
            parameters: Tensor, momentums: Tensor, is_last_step: bool
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

        # Half step for momentums (in the beginning)
        momentums = (
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

        # Full steps
        for i in range(num_leapfrog_steps):
            is_last_step = i == num_leapfrog_steps - 1
            parameters, momentums = leapfrog_step(parameters, momentums, is_last_step)

        # Half step for momentums (in the end)
        momentums = (
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
        return parameters, momentums

    @dataclass
    class MHUpdateState:
        parameters: Tensor
        momentums: Tensor
        next_parameters: Tensor
        next_momentums: Tensor

    def metropolis_hastings_update(state: MHUpdateState) -> tuple[Tensor, bool]:
        potential_energy_func = compile_potential_energy_func()
        kinetic_energy_func = compile_kinetic_energy_func()
        potential_energy = potential_energy_func(state.parameters)
        kinetic_energy = kinetic_energy_func(state.momentums)
        # Negate momentums to make proposal symmetric
        state.next_momentums = -state.next_momentums
        next_potential_energy = potential_energy_func(state.next_parameters)
        next_kinetic_energy = kinetic_energy_func(state.next_momentums)

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
        next_parameters = state.next_parameters
        is_accepted = True
        if rand_uniform_number > acceptance_ratio:
            next_parameters = state.parameters
            is_accepted = False
        return next_parameters, is_accepted

    draw_normalized_momentums_func = compile_draw_normalized_momentums_func(
        initial_parameters
    )

    def one_iteration(parameters: Tensor) -> tuple[Tensor, bool]:
        momentums = draw_normalized_momentums_func()
        # normalized_momentums = draw_normalized_momentums_func()
        # momentums = leapfrog_step_sizes * normalized_momentums
        next_parameters, next_momentums = propose_next_parameters(parameters, momentums)
        mh_update_state = MHUpdateState(
            parameters=parameters,
            momentums=momentums,
            next_parameters=next_parameters,
            next_momentums=next_momentums,
        )
        return metropolis_hastings_update(mh_update_state)

    samples_list: Samples = []
    num_accepted_proposals = 0
    parameters = initial_parameters
    for _ in range(num_total_iterations):
        parameters = parameters.clone().requires_grad_(True)
        parameters, is_accepted = one_iteration(parameters)
        parameters.detach()
        samples_list.append(parameters)
        if is_accepted:
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
