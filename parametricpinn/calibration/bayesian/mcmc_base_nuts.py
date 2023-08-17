import dataclasses
from dataclasses import dataclass
from typing import Callable, TypeAlias, TypeVar

import torch

from parametricpinn.calibration.bayesian.mcmc_base_hamiltonian import (
    Energy,
    Momentums,
    Parameters,
    PotentialEnergyFunc,
    StepSizes,
    _potential_energy_func,
    _sample_normalized_momentums,
    kinetic_energy_func,
)
from parametricpinn.types import Device, Tensor

SliceVariable: TypeAlias = Tensor
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
    potential_energy_func: PotentialEnergyFunc,
) -> LeapfrogStepFunc:
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


SampleSliceVariableFnuc: TypeAlias = Callable[[Parameters, Momentums], SliceVariable]


def _sample_slice_variable(
    potential_energy_func: PotentialEnergyFunc, device: Device
) -> SampleSliceVariableFnuc:
    calculate_negative_hamiltonian = _calculate_negative_hamiltonian(
        potential_energy_func
    )

    def sample_slice_variable(
        parameters: Parameters, momentums: Momentums
    ) -> SliceVariable:
        negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
        unnormalized_joint_probability = torch.exp(negative_hamiltonian)
        return torch.distributions.Uniform(
            low=torch.tensor(0.0, dtype=torch.float64, device=device),
            high=unnormalized_joint_probability,
        ).sample()

    return sample_slice_variable


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
IsErrorTooLargeFunc: TypeAlias = Callable[[Parameters, Momentums, SliceVariable], bool]


def _is_error_too_large(
    potential_energy_func: PotentialEnergyFunc, delta_error: DeltaError, device: Device
) -> IsErrorTooLargeFunc:
    calculate_negative_hamiltonian = _calculate_negative_hamiltonian(
        potential_energy_func
    )

    def is_error_too_large(
        parameters: Parameters, momentums: Momentums, slice_variable: SliceVariable
    ) -> bool:
        negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
        return bool(negative_hamiltonian - torch.log(slice_variable) < -delta_error)

    return is_error_too_large


IsStateInSliceFunc: TypeAlias = Callable[[Parameters, Momentums, SliceVariable], bool]


def _is_state_in_slice(
    potential_energy_func: PotentialEnergyFunc, device: Device
) -> IsStateInSliceFunc:
    calculate_negative_hamiltonian = _calculate_negative_hamiltonian(
        potential_energy_func
    )

    def is_state_in_slice(
        parameters: Parameters, momentums: Momentums, slice_variable: SliceVariable
    ) -> bool:
        negative_hamiltonian = calculate_negative_hamiltonian(parameters, momentums)
        unnormalized_joint_probability = torch.exp(negative_hamiltonian)
        return bool(slice_variable <= unnormalized_joint_probability)

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


def log_bernoulli(log_probability: Tensor) -> bool:
    if torch.isnan(log_probability):
        raise FloatingPointError("log_probability can't be nan.")
    return bool(torch.log(torch.rand(1)) < log_probability)
