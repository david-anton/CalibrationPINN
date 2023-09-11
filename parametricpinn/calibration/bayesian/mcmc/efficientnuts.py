import random
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import Likelihood
from parametricpinn.calibration.bayesian.mcmc.base import (
    MCMCOutput,
    Samples,
    _log_unnormalized_posterior,
    expand_num_iterations,
    log_bernoulli,
    postprocess_samples,
    remove_burn_in_phase,
)
from parametricpinn.calibration.bayesian.mcmc.base_nuts import (
    Direction,
    LogSliceVariable,
    Momentums,
    Parameters,
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
    _sample_log_slice_variable,
    _sample_normalized_momentums,
    is_max_tree_depth_reached,
    keep_minus_from_old_tree,
    keep_plus_from_old_tree,
)
from parametricpinn.calibration.bayesian.mcmc.config import MCMCConfig
from parametricpinn.calibration.bayesian.prior import Prior
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.types import Device, NPArray, Tensor

CandidateSetSize: TypeAlias = Tensor

MCMCEfficientNUTSFunc: TypeAlias = Callable[
    [
        Likelihood,
        Prior,
        Parameters,
        StepSizes,
        int,
        int,
        TreeDepth,
        Device,
    ],
    MCMCOutput,
]


@dataclass
class EfficientNUTSConfig(MCMCConfig):
    leapfrog_step_sizes: Tensor
    max_tree_depth: TreeDepth


@dataclass
class EfficientTree(Tree):
    parameters_candidate: Parameters
    candidate_set_size: CandidateSetSize
    is_terminated: TerminationFlag


def mcmc_efficientnuts(
    likelihood: Likelihood,
    prior: Prior,
    initial_parameters: Parameters,
    leapfrog_step_sizes: StepSizes,
    num_iterations: int,
    num_burn_in_iterations: int,
    max_tree_depth: TreeDepth,
    device: Device,
) -> MCMCOutput:
    step_sizes = leapfrog_step_sizes
    directions = [-1, 1]
    max_delta_error = torch.tensor(1000.0, device=device)
    num_total_iterations = expand_num_iterations(num_iterations, num_burn_in_iterations)

    log_unnorm_posterior = _log_unnormalized_posterior(likelihood, prior)
    grad_log_unnorm_posterior = _grad_log_unnormalized_posterior(likelihood, prior)
    potential_energy_func = _potential_energy_func(log_unnorm_posterior)
    grad_potential_energy_func = _grad_potential_energy_func(grad_log_unnorm_posterior)
    sample_momentums = _sample_normalized_momentums(initial_parameters, device)
    sample_log_slice_variable = _sample_log_slice_variable(
        potential_energy_func, device
    )
    leapfrog_step = _leapfrog_step(grad_potential_energy_func)
    is_distance_decreasing = _is_distance_decreasing(device)
    is_error_too_large = _is_error_too_large(potential_energy_func, max_delta_error)
    is_state_in_slice = _is_state_in_slice(potential_energy_func)

    def efficient_nuts_sampler(parameters: Parameters) -> Parameters:
        def update_parameters_canditate_after_doubling(
            tree: EfficientTree,
            tree_s1: EfficientTree,
        ) -> Parameters:
            size = tree.candidate_set_size
            size_s1 = tree_s1.candidate_set_size
            log_acceptance_probability = torch.log(size_s1) - torch.log(size)
            parameters_candidate = tree.parameters_candidate
            if log_bernoulli(log_acceptance_probability, device):
                parameters_candidate = tree_s1.parameters_candidate
            return parameters_candidate

        def update_parameters_canditate_in_subtrees(
            tree_s1: EfficientTree, tree_s2: EfficientTree
        ) -> Parameters:
            size_s1 = tree_s1.candidate_set_size
            size_s2 = tree_s2.candidate_set_size
            log_acceptance_probability = torch.log(size_s2) - torch.log(
                size_s1 + size_s2
            )
            parameters_candidate = tree_s1.parameters_candidate
            if log_bernoulli(log_acceptance_probability, device):
                parameters_candidate = tree_s2.parameters_candidate
            return parameters_candidate

        def base_step(
            parameters: Parameters,
            momentums: Momentums,
            log_slice_variable: LogSliceVariable,
            direction: Direction,
            step_sizes: StepSizes,
        ) -> EfficientTree:
            """Take one leapfrog step in direction 'direction'."""
            parameters_s1, momentums_s1 = leapfrog_step(
                parameters, momentums, direction * step_sizes
            )
            candidate_set_size = torch.tensor(0, dtype=torch.int16, device=device)
            if is_state_in_slice(parameters_s1, momentums_s1, log_slice_variable):
                candidate_set_size = torch.tensor(1, dtype=torch.int16, device=device)
            is_terminated_s1 = is_error_too_large(
                parameters_s1, momentums_s1, log_slice_variable
            )
            return EfficientTree(
                parameters_m=parameters_s1,
                momentums_m=momentums_s1,
                parameters_p=parameters_s1,
                momentums_p=momentums_s1,
                parameters_candidate=parameters_s1,
                candidate_set_size=candidate_set_size,
                is_terminated=is_terminated_s1,
            )

        def build_subtree(
            parameters: Parameters,
            momentums: Momentums,
            log_slice_variable: LogSliceVariable,
            direction: Direction,
            tree_depth: TreeDepth,
            step_sizes: StepSizes,
        ) -> EfficientTree:
            """Build left and right subtrees."""
            tree_depth = tree_depth - 1
            tree_s1 = build_tree(
                parameters=parameters,
                momentums=momentums,
                log_slice_variable=log_slice_variable,
                direction=direction,
                tree_depth=tree_depth,
                step_sizes=step_sizes,
            )
            if not tree_s1.is_terminated:
                if direction == -1:
                    tree_s2 = keep_plus_from_old_tree(
                        new_tree=build_tree(
                            parameters=tree_s1.parameters_m,
                            momentums=tree_s1.momentums_m,
                            log_slice_variable=log_slice_variable,
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
                            log_slice_variable=log_slice_variable,
                            direction=direction,
                            tree_depth=tree_depth,
                            step_sizes=step_sizes,
                        ),
                        old_tree=tree_s1,
                    )

                parameters_candidate = tree_s1.parameters_candidate
                parameters_candidate = update_parameters_canditate_in_subtrees(
                    tree_s1=tree_s1, tree_s2=tree_s2
                )
                candidate_set_size = (
                    tree_s1.candidate_set_size + tree_s2.candidate_set_size
                )
                is_terminated = tree_s2.is_terminated or is_distance_decreasing(tree_s2)
                return EfficientTree(
                    parameters_m=tree_s2.parameters_m,
                    momentums_m=tree_s2.momentums_m,
                    parameters_p=tree_s2.parameters_p,
                    momentums_p=tree_s2.momentums_p,
                    parameters_candidate=parameters_candidate,
                    candidate_set_size=candidate_set_size,
                    is_terminated=is_terminated,
                )
            return tree_s1

        def build_tree(
            parameters: Parameters,
            momentums: Momentums,
            log_slice_variable: LogSliceVariable,
            direction: Direction,
            tree_depth: TreeDepth,
            step_sizes: StepSizes,
        ) -> EfficientTree:
            if tree_depth == 0:
                return base_step(
                    parameters=parameters,
                    momentums=momentums,
                    log_slice_variable=log_slice_variable,
                    direction=direction,
                    step_sizes=step_sizes,
                )
            else:
                return build_subtree(
                    parameters=parameters,
                    momentums=momentums,
                    log_slice_variable=log_slice_variable,
                    direction=direction,
                    tree_depth=tree_depth,
                    step_sizes=step_sizes,
                )

        momentums = sample_momentums()
        log_slice_variable = sample_log_slice_variable(parameters, momentums)
        tree = EfficientTree(
            parameters_m=parameters,
            momentums_m=momentums,
            parameters_p=parameters,
            momentums_p=momentums,
            parameters_candidate=parameters,
            candidate_set_size=torch.tensor(1, dtype=torch.int16, device=device),
            is_terminated=False,
        )
        max_tree_depth_reached = False
        tree_depth = 0

        while not tree.is_terminated and not max_tree_depth_reached:
            direction = random.choice(directions)

            if direction == -1:
                tree_s1 = keep_plus_from_old_tree(
                    new_tree=build_tree(
                        parameters=tree.parameters_m,
                        momentums=tree.momentums_m,
                        log_slice_variable=log_slice_variable,
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
                        log_slice_variable=log_slice_variable,
                        direction=direction,
                        tree_depth=tree_depth,
                        step_sizes=step_sizes,
                    ),
                    old_tree=tree,
                )

            parameters_candidate = tree.parameters_candidate
            if not tree_s1.is_terminated:
                parameters_candidate = update_parameters_canditate_after_doubling(
                    tree=tree,
                    tree_s1=tree_s1,
                )
            candidate_set_size = tree.candidate_set_size + tree_s1.candidate_set_size
            is_terminated = tree_s1.is_terminated or is_distance_decreasing(tree_s1)

            tree = EfficientTree(
                parameters_m=tree_s1.parameters_m,
                momentums_m=tree_s1.momentums_m,
                parameters_p=tree_s1.parameters_p,
                momentums_p=tree_s1.momentums_p,
                parameters_candidate=parameters_candidate,
                candidate_set_size=candidate_set_size,
                is_terminated=is_terminated,
            )
            max_tree_depth_reached = is_max_tree_depth_reached(
                tree_depth, max_tree_depth
            )
            tree_depth += 1

        return parameters_candidate

    def one_iteration(parameters: Tensor) -> Tensor:
        return efficient_nuts_sampler(parameters)

    samples_list: Samples = []
    parameters = initial_parameters
    for i in range(num_total_iterations):
        parameters = (
            parameters.clone().to(device).type(torch.float64).requires_grad_(True)
        )
        parameters = one_iteration(parameters)
        parameters.detach()
        samples_list.append(parameters)

    samples_list = remove_burn_in_phase(
        sample_list=samples_list, num_burn_in_iterations=num_burn_in_iterations
    )
    moments, samples = postprocess_samples(samples_list=samples_list)

    return moments, samples
