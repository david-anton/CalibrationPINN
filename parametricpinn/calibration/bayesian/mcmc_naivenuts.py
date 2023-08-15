import dataclasses
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

Parameters: TypeAlias = Tensor
Momentums: TypeAlias = Tensor
MomentumsDistribution = Union[TorchUniNormalDist, TorchMultiNormalDist]
DrawMomentumsFunc: TypeAlias = Callable[[], Momentums]
SliceVariable: TypeAlias = Tensor
Energy: TypeAlias = Tensor
Hamiltonian: TypeAlias = Energy
StepSizes: TypeAlias = Tensor
Direction: TypeAlias = int
CandidateStates: TypeAlias = list[tuple[Parameters, Momentums]]
TerminationFlag: TypeAlias = bool
TreeDepth: TypeAlias = int
MCMCNaiveNUTSFunc: TypeAlias = Callable[
    [
        tuple[str, ...],
        tuple[float, ...],
        LikelihoodFunc,
        PriorFunc,
        Parameters,
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
class NaiveNUTSConfig(MCMCConfig):
    leapfrog_step_sizes: Tensor


@dataclass
class Tree:
    parameters_m: Parameters
    momentums_m: Momentums
    parameters_p: Parameters
    momentums_p: Momentums
    candidate_states: CandidateStates
    is_terminated: TerminationFlag


def mcmc_naivenuts(
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

    def potential_energy_func(parameters: Parameters) -> Energy:
        return -torch.log(unnormalized_posterior(parameters))

    def kinetic_energy_func(momentums: Momentums) -> Energy:
        return 1 / 2 * torch.sum(torch.pow(momentums, 2))

    def compile_draw_normalized_momentums_func(
        parameters: Parameters,
    ) -> DrawMomentumsFunc:
        def compile_momentum_distribution() -> MomentumsDistribution:
            if parameters.size() == (1,):
                mean = torch.tensor([0.0], dtype=torch.float64, device=device)
                standard_deviation = torch.tensor(
                    [1.0], dtype=torch.float64, device=device
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

    def naive_nuts_sampler(parameters: Parameters) -> Parameters:
        def calculate_negative_hamiltonian(
            parameters: Parameters, momentums: Momentums
        ) -> Hamiltonian:
            potential_energy = potential_energy_func(parameters)
            kinetic_energy = kinetic_energy_func(momentums)
            return -1 * (potential_energy + kinetic_energy)

        def sample_slice_variable(
            parameters: Parameters, momentums: Momentums
        ) -> SliceVariable:
            negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
            return torch.distributions.Uniform(
                low=torch.tensor(0.0, dtype=torch.float64, device=device),
                high=negative_hamiltonian,
            ).sample()

        def is_distance_decreasing(tree: Tree) -> TerminationFlag:
            parameters_delta = tree.parameters_p - tree.parameters_m
            distance_progress_m = torch.matmul(parameters_delta, tree.momentums_m)
            distance_progress_p = torch.matmul(parameters_delta, tree.momentums_p)
            zero = torch.tensor(0.0, device=device)
            if distance_progress_m < zero or distance_progress_p < zero:
                return True
            return False

        def is_error_too_large(
            parameters: Parameters, momentums: Momentums, slice_variable: SliceVariable
        ) -> bool:
            negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
            return bool(negative_hamiltonian - torch.log(slice_variable) < -delta_error)

        def is_state_in_slice(
            parameters: Parameters, momentums: Momentums, slice_variable: SliceVariable
        ) -> bool:
            negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
            return bool(slice_variable <= negative_hamiltonian)

        def keep_plus_from_old_tree(new_tree: Tree, old_tree: Tree) -> Tree:
            new_tree_copy = dataclasses.replace(new_tree)
            new_tree_copy.parameters_p = old_tree.parameters_p
            new_tree_copy.momentums_p = old_tree.momentums_p
            return new_tree_copy

        def keep_minus_from_old_tree(new_tree: Tree, old_tree: Tree) -> Tree:
            new_tree_copy = dataclasses.replace(new_tree)
            new_tree_copy.parameters_m = old_tree.parameters_m
            new_tree_copy.momentums_m = old_tree.momentums_m
            return new_tree_copy

        def build_tree_base_case(
            parameters: Parameters,
            momentums: Momentums,
            slice_variable: SliceVariable,
            direction: Direction,
            step_sizes: StepSizes,
        ) -> Tree:
            """Take one leapfrog step in direction 'direction'."""
            parameters_s1, momentums_s1 = leapfrog_step(
                parameters, momentums, direction * step_sizes
            )
            candidate_states = []
            if is_state_in_slice(parameters_s1, momentums_s1, slice_variable):
                candidate_states.append((parameters_s1, momentums_s1))
            is_terminated = is_error_too_large(
                parameters_s1, momentums_s1, slice_variable
            )
            return Tree(
                parameters_m=parameters_s1,
                momentums_m=momentums_s1,
                parameters_p=parameters_s1,
                momentums_p=momentums_s1,
                candidate_states=candidate_states,
                is_terminated=is_terminated,
            )

        def build_tree_recursive_case(
            parameters: Parameters,
            momentums: Momentums,
            slice_variable: SliceVariable,
            direction: Direction,
            tree_depth: TreeDepth,
            step_sizes: StepSizes,
        ) -> Tree:
            """Build left and right subtrees."""
            tree_depth = tree_depth - 1
            tree_s1 = build_tree(
                parameters=parameters,
                momentums=momentums,
                slice_variable=slice_variable,
                direction=direction,
                tree_depth=tree_depth,
                step_sizes=step_sizes,
            )
            if direction == -1:
                tree_s2 = keep_plus_from_old_tree(
                    new_tree=build_tree(
                        parameters=tree_s1.parameters_m,
                        momentums=tree_s1.momentums_m,
                        slice_variable=slice_variable,
                        direction=direction,
                        tree_depth=tree_depth,
                        step_sizes=step_sizes,
                    ),
                    old_tree=tree_s1,
                )
            else:
                tree_s2 = keep_minus_from_old_tree(
                    new_tree=build_tree(
                        parameters=tree_s1.parameters_p,
                        momentums=tree_s1.momentums_p,
                        slice_variable=slice_variable,
                        direction=direction,
                        tree_depth=tree_depth,
                        step_sizes=step_sizes,
                    ),
                    old_tree=tree_s1,
                )

            candidate_states = tree_s1.candidate_states
            candidate_states.extend(tree_s2.candidate_states)
            is_terminated = (
                tree_s1.is_terminated
                or tree_s2.is_terminated
                or is_distance_decreasing(tree_s2)
            )
            return Tree(
                parameters_m=tree_s2.parameters_m,
                momentums_m=tree_s2.momentums_m,
                parameters_p=tree_s2.parameters_p,
                momentums_p=tree_s2.momentums_p,
                candidate_states=candidate_states,
                is_terminated=is_terminated,
            )

        def build_tree(
            parameters: Parameters,
            momentums: Momentums,
            slice_variable: SliceVariable,
            direction: Direction,
            tree_depth: TreeDepth,
            step_sizes: StepSizes,
        ) -> Tree:
            if tree_depth == 0:
                return build_tree_base_case(
                    parameters=parameters,
                    momentums=momentums,
                    slice_variable=slice_variable,
                    direction=direction,
                    step_sizes=step_sizes,
                )
            else:
                return build_tree_recursive_case(
                    parameters=parameters,
                    momentums=momentums,
                    slice_variable=slice_variable,
                    direction=direction,
                    tree_depth=tree_depth,
                    step_sizes=step_sizes,
                )

        sample_normalized_momentums = compile_draw_normalized_momentums_func(
            initial_parameters
        )
        step_sizes = leapfrog_step_sizes
        directions = [-1, 1]
        delta_error = torch.tensor(1000.0, device=device)

        momentums = sample_normalized_momentums()
        slice_variable = sample_slice_variable(parameters, momentums)
        tree = Tree(
            parameters_m=parameters,
            momentums_m=momentums,
            parameters_p=parameters,
            momentums_p=momentums,
            candidate_states=[(parameters, momentums)],
            is_terminated=False,
        )
        tree_depth = 0

        while not tree.is_terminated:
            direction = random.choice(directions)

            if direction == -1:
                tree_s1 = keep_plus_from_old_tree(
                    new_tree=build_tree(
                        parameters=tree.parameters_m,
                        momentums=tree.momentums_m,
                        slice_variable=slice_variable,
                        direction=direction,
                        tree_depth=tree_depth,
                        step_sizes=step_sizes,
                    ),
                    old_tree=tree,
                )
            else:
                tree_s1 = keep_minus_from_old_tree(
                    new_tree=build_tree(
                        parameters=tree.parameters_p,
                        momentums=tree.momentums_p,
                        slice_variable=slice_variable,
                        direction=direction,
                        tree_depth=tree_depth,
                        step_sizes=step_sizes,
                    ),
                    old_tree=tree,
                )

            candidate_states = tree.candidate_states
            if not tree_s1.is_terminated:
                candidate_states.extend(tree_s1.candidate_states)
            is_terminated = tree_s1.is_terminated or is_distance_decreasing(tree_s1)

            tree = Tree(
                parameters_m=tree_s1.parameters_m,
                momentums_m=tree_s1.momentums_m,
                parameters_p=tree_s1.parameters_p,
                momentums_p=tree_s1.momentums_p,
                candidate_states=candidate_states,
                is_terminated=is_terminated,
            )
            tree_depth += 1

        parameters, momentums = random.choice(tree.candidate_states)
        # print(f"Step length: {2**tree_depth}")
        return parameters

    def one_iteration(parameters: Tensor) -> Tensor:
        return naive_nuts_sampler(parameters)

    samples_list: Samples = []
    parameters = initial_parameters
    for i in range(num_total_iterations):
        parameters = parameters.clone().type(torch.float64).requires_grad_(True)
        parameters = one_iteration(parameters)
        parameters.detach()
        samples_list.append(parameters)
        # print(f"Iteration: {i}")

    samples_list = remove_burn_in_phase(
        sample_list=samples_list, num_burn_in_iterations=num_burn_in_iterations
    )
    moments, samples = postprocess_samples(
        samples_list=samples_list,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        mcmc_algorithm="naivenuts_mcmc",
        output_subdir=output_subdir,
        project_directory=project_directory,
    )

    return moments, samples
