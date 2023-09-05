import dataclasses
from dataclasses import dataclass
from typing import Callable, TypeAlias, TypeVar

import torch

from parametricpinn.calibration.bayesian.mcmc.base_hamiltonian import (
    Energy,
    GradPotentialEnergyFunc,
    Momentums,
    Parameters,
    PotentialEnergyFunc,
    StepSizes,
    _grad_log_unnormalized_posterior,
    _grad_potential_energy_func,
    _potential_energy_func,
    _sample_normalized_momentums,
    kinetic_energy_func,
)
from parametricpinn.types import Device, Tensor

LogSliceVariable: TypeAlias = Tensor
Hamiltonian: TypeAlias = Energy
Direction: TypeAlias = int
TerminationFlag: TypeAlias = bool
TreeDepth: TypeAlias = int


@dataclass
class Tree:
    parameters_m: Parameters
    momentums_m: Momentums
    parameters_p: Parameters
    momentums_p: Momentums


LeapfrogStepFunc: TypeAlias = Callable[
    [Parameters, Momentums, StepSizes], tuple[Parameters, Momentums]
]


def _leapfrog_step(
    grad_potential_energy_func: GradPotentialEnergyFunc,
) -> LeapfrogStepFunc:
    def leapfrog_step(
        parameters: Parameters, momentums: Momentums, step_sizes: StepSizes
    ) -> tuple[Parameters, Momentums]:
        def half_momentums_step(
            parameters: Parameters, momentums: Momentums, step_sizes: StepSizes
        ) -> Momentums:
            return momentums - step_sizes / 2 * grad_potential_energy_func(parameters)

        def full_parameter_step(
            parameters: Parameters, momentums: Momentums, step_sizes: StepSizes
        ) -> Parameters:
            return parameters + step_sizes * momentums

        momentums = half_momentums_step(parameters, momentums, step_sizes)
        parameters = full_parameter_step(parameters, momentums, step_sizes)
        momentums = half_momentums_step(parameters, momentums, step_sizes)
        return parameters, momentums

    return leapfrog_step


CalculateNegativeHamiltonianFunc: TypeAlias = Callable[
    [Parameters, Momentums], Hamiltonian
]


def _calculate_negative_hamiltonian(
    potential_energy_func: PotentialEnergyFunc,
) -> CalculateNegativeHamiltonianFunc:
    def calculate_negative_hamiltonian(
        parameters: Parameters, momentums: Momentums
    ) -> Hamiltonian:
        potential_energy = potential_energy_func(parameters)
        kinetic_energy = kinetic_energy_func(momentums)
        return -1 * (potential_energy + kinetic_energy)

    return calculate_negative_hamiltonian


SampleLogSliceVariableFunc: TypeAlias = Callable[
    [Parameters, Momentums], LogSliceVariable
]


def _sample_log_slice_variable(
    potential_energy_func: PotentialEnergyFunc, device: Device
) -> SampleLogSliceVariableFunc:
    calculate_negative_hamiltonian = _calculate_negative_hamiltonian(
        potential_energy_func
    )

    def sample_log_slice_variable(
        parameters: Parameters, momentums: Momentums
    ) -> LogSliceVariable:
        negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
        # The exponential distribution of the logarithmized slice variable follows from a logarithmic transformation
        # of the uniform distributed slice variable u ~ Uniform(0, exp(-Hamiltonain)). The logarithmic transformation
        # is applied to avoid floating point undeflow.
        # See page 712 in "M. Neal, Slice Sampling, 2003".
        return (
            negative_hamiltonian
            - torch.distributions.Exponential(
                rate=torch.tensor(1.0, dtype=torch.float64, device=device)
            ).sample()
        )

    return sample_log_slice_variable


IsDistanceDecreasingFunc: TypeAlias = Callable[[Tree], TerminationFlag]


def _is_distance_decreasing(device: Device) -> IsDistanceDecreasingFunc:
    def is_distance_decreasing(tree: Tree) -> TerminationFlag:
        parameters_delta = tree.parameters_p - tree.parameters_m
        distance_progress_m = torch.matmul(parameters_delta, tree.momentums_m)
        distance_progress_p = torch.matmul(parameters_delta, tree.momentums_p)
        zero = torch.tensor(0.0, device=device)
        if distance_progress_m < zero or distance_progress_p < zero:
            return True
        return False

    return is_distance_decreasing


DeltaError: TypeAlias = Tensor
IsErrorTooLargeFunc: TypeAlias = Callable[
    [Parameters, Momentums, LogSliceVariable], bool
]


def _is_error_too_large(
    potential_energy_func: PotentialEnergyFunc, delta_error: DeltaError
) -> IsErrorTooLargeFunc:
    calculate_negative_hamiltonian = _calculate_negative_hamiltonian(
        potential_energy_func
    )

    def is_error_too_large(
        parameters: Parameters,
        momentums: Momentums,
        log_slice_variable: LogSliceVariable,
    ) -> bool:
        negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
        return bool(negative_hamiltonian - log_slice_variable <= -delta_error)

    return is_error_too_large


IsStateInSliceFunc: TypeAlias = Callable[
    [Parameters, Momentums, LogSliceVariable], bool
]


def _is_state_in_slice(
    potential_energy_func: PotentialEnergyFunc,
) -> IsStateInSliceFunc:
    calculate_negative_hamiltonian = _calculate_negative_hamiltonian(
        potential_energy_func
    )

    def is_state_in_slice(
        parameters: Parameters,
        momentums: Momentums,
        log_slice_variable: LogSliceVariable,
    ) -> bool:
        negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
        return bool(log_slice_variable <= negative_hamiltonian)

    return is_state_in_slice


TreeType = TypeVar("TreeType", bound=Tree)


def keep_plus_from_old_tree(new_tree: TreeType, old_tree: TreeType) -> TreeType:
    new_tree_copy = dataclasses.replace(new_tree)
    new_tree_copy.parameters_p = old_tree.parameters_p
    new_tree_copy.momentums_p = old_tree.momentums_p
    return new_tree_copy


def keep_minus_from_old_tree(new_tree: TreeType, old_tree: TreeType) -> TreeType:
    new_tree_copy = dataclasses.replace(new_tree)
    new_tree_copy.parameters_m = old_tree.parameters_m
    new_tree_copy.momentums_m = old_tree.momentums_m
    return new_tree_copy


def is_max_tree_depth_reached(tree_depth: TreeDepth, max_tree_depth: TreeDepth) -> bool:
    return tree_depth >= max_tree_depth
