import random
from dataclasses import dataclass
from typing import Callable, TypeAlias, Union

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.calibration.bayesian.mcmc_base import (
    Samples,
    compile_unnnormalized_posterior,
    expand_num_iterations,
    postprocess_samples,
    remove_burn_in_phase,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.prior import PriorFunc
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
        PriorFunc,
        Tensor,
        Tensor,
        int,
        int,
        str,
        ProjectDirectory,
        Device,
    ],
    tuple[MomentsMultivariateNormal, NPArray],
]
Parameters: TypeAlias = Tensor
Momentums: TypeAlias = Tensor
MomentumsDistribution = Union[TorchUniNormalDist, TorchMultiNormalDist]
DrawMomentumsFunc: TypeAlias = Callable[[], Momentums]
StepSizes: TypeAlias = Tensor
Direction: TypeAlias = int
Directions: TypeAlias = list[Direction]
SliceVariable: TypeAlias = Tensor
TerminationFlag: TypeAlias = int
CandidateStates: TypeAlias = list[tuple[Parameters, Momentums]]
Tree: TypeAlias = tuple[
    Parameters, Momentums, Parameters, Momentums, CandidateStates, int
]
TreeDepth: TypeAlias = int


@dataclass
class EfficientNUTSConfig(MCMCConfig):
    leapfrog_step_sizes: Tensor


def mcmc_efficientnuts(
    parameter_names: tuple[str, ...],
    true_parameters: tuple[float, ...],
    likelihood: LikelihoodFunc,
    prior: PriorFunc,
    initial_parameters: Parameters,
    leapfrog_step_sizes: StepSizes,
    num_iterations: int,
    num_burn_in_iterations: int,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    num_total_iterations = expand_num_iterations(num_iterations, num_burn_in_iterations)
    unnormalized_posterior = compile_unnnormalized_posterior(likelihood, prior)

    def potential_energy_func(parameters: Parameters) -> Tensor:
        return -torch.log(unnormalized_posterior(parameters))

    def kinetic_energy_func(momentums: Momentums) -> Tensor:
        return 1 / 2 * torch.sum(torch.pow(momentums, 2))

    def compile_draw_normalized_momentums_func(
        parameters: Parameters,
    ) -> DrawMomentumsFunc:
        def compile_momentum_distribution() -> MomentumsDistribution:
            if parameters.size() == (1,):
                mean = torch.tensor(0.0, dtype=torch.float64, device=device)
                standard_deviation = torch.tensor(
                    1.0, dtype=torch.float64, device=device
                )
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

        def draw_momentums_func() -> Momentums:
            momentums = momentums_dist.sample()
            return momentums.requires_grad_(True)

        return draw_momentums_func

    draw_normalized_momentums = compile_draw_normalized_momentums_func(
        initial_parameters
    )

    def draw_slice_variable(
        parameters: Parameters, momentums: Momentums
    ) -> SliceVariable:
        potential_energy = potential_energy_func(parameters)
        kinetic_energy = kinetic_energy_func(momentums)
        negative_hamiltonian = -1 * (potential_energy + kinetic_energy)
        return torch.distributions.Uniform(
            low=torch.tensor(0.0, device=device), high=negative_hamiltonian
        ).sample()

    def leapfrog_step(
        parameters: Parameters, momentums: Momentums, step_sizes: StepSizes
    ) -> tuple[Parameters, Momentums]:
        def half_momentums_step(
            parameters: Parameters, momentums: Momentums, step_sizes: StepSizes
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

        def full_parameter_step(
            parameters: Parameters, momentums: Momentums, step_sizes: StepSizes
        ) -> Parameters:
            return parameters + step_sizes * momentums

        momentums = half_momentums_step(parameters, momentums, step_sizes)
        parameters = full_parameter_step(parameters, momentums, step_sizes)
        momentums = half_momentums_step(parameters, momentums, step_sizes)

        return parameters, momentums

    def efficient_nuts_sampler(parameters: Parameters) -> Parameters:
        momentums = draw_normalized_momentums()
        slice_variable = draw_slice_variable(parameters, momentums)
        parameters_m = parameters_p = parameters
        momentums_m = momentums_p = momentums
        step_sizes = leapfrog_step_sizes
        directions = [-1, 1]
        tree_depth = 0
        is_terminated = 1  # 1=False, 0=True
        candidate_states = [(parameters, momentums)]

        def build_tree(
            parameters: Parameters,
            momentums: Momentums,
            slice_variable: SliceVariable,
            direction: Direction,
            tree_depth: TreeDepth,
            step_sizes: StepSizes,
        ) -> Tree:
            pass

        while is_terminated == 1:
            direction = random.choice(directions)

            if direction == -1:
                (
                    parameters_m,
                    momentums_m,
                    _,
                    _,
                    candidate_states_i,
                    is_terminated_i,
                ) = build_tree(
                    parameters=parameters_m,
                    momentums=momentums_m,
                    slice_variable=slice_variable,
                    direction=direction,
                    tree_depth=tree_depth,
                    step_sizes=step_sizes,
                )
            else:
                (
                    _,
                    _,
                    parameters_p,
                    momentums_p,
                    candidate_states_i,
                    is_terminated_i,
                ) = build_tree(
                    parameters=parameters_p,
                    momentums=momentums_p,
                    slice_variable=slice_variable,
                    direction=direction,
                    tree_depth=tree_depth,
                    step_sizes=step_sizes,
                )
            if is_terminated_i == 1:
                candidate_states.extend(candidate_states_i)

        def check_distance_criterion(
            parameters_m: Parameters,
            momentums_m: Momentums,
            parameters_p: Parameters,
            momentums_p: Momentums,
        ) -> TerminationFlag:
            # parameters_change = parameters_p - parameters_m
            # distance_progress_m = torch.matmul(parameters_change, momentums_m)
            # distance_progress_p = torch.matmul(parameters_change, momentums_p)
            # if distance_progress_m >= torch.tensor(
            #     0.0, device=device
            # ) and distance_progress_p >= torch.tensor(0.0, device=device):
            #     return 1
            # else:
            #     return 0

        return parameters

    ################################################################
    def one_iteration(parameters: Tensor) -> Tensor:
        return efficient_nuts_sampler(parameters)

    samples_list: Samples = []
    parameters = initial_parameters
    for i in range(num_total_iterations):
        parameters = parameters.clone().type(torch.float64).requires_grad_(True)
        parameters = one_iteration(parameters)
        parameters.detach()
        samples_list.append(parameters)

    samples_list = remove_burn_in_phase(
        sample_list=samples_list, num_burn_in_iterations=num_burn_in_iterations
    )
    moments, samples = postprocess_samples(
        samples_list=samples_list,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        mcmc_algorithm="efficientnuts_mcmc",
        output_subdir=output_subdir,
        project_directory=project_directory,
    )

    return moments, samples
