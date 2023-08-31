import random
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import Likelihood
from parametricpinn.calibration.bayesian.mcmc.base import (
    Samples,
    _log_unnormalized_posterior,
    expand_num_iterations,
    postprocess_samples,
    remove_burn_in_phase,
)
from parametricpinn.calibration.bayesian.mcmc.base_nuts import (
    Direction,
    Momentums,
    Parameters,
    SliceVariable,
    StepSizes,
    TerminationFlag,
    Tree,
    TreeDepth,
    _grad_log_unnormalized_posterior,
    _grad_potential_energy_func,
    _is_distance_decreasing,
    _is_error_too_large,
    _is_state_in_slice,
    _leapfrog_step,
    _potential_energy_func,
    _sample_normalized_momentums,
    _sample_slice_variable,
    keep_minus_from_old_tree,
    keep_plus_from_old_tree,
)
from parametricpinn.calibration.bayesian.mcmc.config import MCMCConfig
from parametricpinn.calibration.bayesian.prior import Prior
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.types import Device, NPArray, Tensor

CandidateStates: TypeAlias = list[tuple[Parameters, Momentums]]

MCMCNaiveNUTSFunc: TypeAlias = Callable[
    [
        Likelihood,
        Prior,
        Parameters,
        StepSizes,
        int,
        int,
        Device,
    ],
    tuple[MomentsMultivariateNormal, NPArray],
]


@dataclass
class NaiveNUTSConfig(MCMCConfig):
    leapfrog_step_sizes: Tensor


@dataclass
class NaiveTree(Tree):
    candidate_states: CandidateStates
    is_terminated: TerminationFlag


def mcmc_naivenuts(
    likelihood: Likelihood,
    prior: Prior,
    initial_parameters: Parameters,
    leapfrog_step_sizes: StepSizes,
    num_iterations: int,
    num_burn_in_iterations: int,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    step_sizes = leapfrog_step_sizes
    directions = [-1, 1]
    delta_error = torch.tensor(1000.0, device=device)
    num_total_iterations = expand_num_iterations(num_iterations, num_burn_in_iterations)

    log_unnorm_posterior = _log_unnormalized_posterior(likelihood, prior)
    grad_log_unnorm_posterior = _grad_log_unnormalized_posterior(likelihood, prior)
    potential_energy_func = _potential_energy_func(log_unnorm_posterior)
    grad_potential_energy_func = _grad_potential_energy_func(grad_log_unnorm_posterior)
    sample_momentums = _sample_normalized_momentums(initial_parameters, device)
    sample_slice_variable = _sample_slice_variable(potential_energy_func, device)
    leapfrog_step = _leapfrog_step(grad_potential_energy_func)
    is_distance_decreasing = _is_distance_decreasing(device)
    is_error_too_large = _is_error_too_large(potential_energy_func, delta_error)
    is_state_in_slice = _is_state_in_slice(potential_energy_func)

    def naive_nuts_sampler(parameters: Parameters) -> Parameters:
        def build_tree_base_case(
            parameters: Parameters,
            momentums: Momentums,
            slice_variable: SliceVariable,
            direction: Direction,
            step_sizes: StepSizes,
        ) -> NaiveTree:
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
            return NaiveTree(
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
        ) -> NaiveTree:
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
            return NaiveTree(
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
        ) -> NaiveTree:
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

        momentums = sample_momentums()
        slice_variable = sample_slice_variable(parameters, momentums)
        tree = NaiveTree(
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

            tree = NaiveTree(
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
    moments, samples = postprocess_samples(samples_list=samples_list)

    return moments, samples
