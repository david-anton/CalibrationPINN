from itertools import groupby
from typing import Any, TypeAlias, Union

import torch

from parametricpinn.errors import ParameterSamplingError
from parametricpinn.types import Device, Tensor

FloatList: TypeAlias = list[float]
IntList: TypeAlias = list[int]
Lists: TypeAlias = Union[FloatList, IntList]


def sample_uniform_grid(
    min_parameters: FloatList,
    max_parameters: FloatList,
    num_steps: IntList,
    device: Device,
) -> Tensor:
    _validate_equal_length_of_lists([min_parameters, max_parameters, num_steps])
    num_dimensions = len(num_steps)
    parameter_steps = [
        torch.linspace(
            min_parameters[i],
            max_parameters[i],
            steps=num_steps[i],
        )
        for i in range(num_dimensions)
    ]
    parameters = torch.cartesian_prod(*parameter_steps).to(device).requires_grad_(True)
    if num_dimensions == 1:
        parameters = _reshape_parameters_for_one_dimension(parameters)
    return parameters


def sample_quasirandom_sobol(
    min_parameters: FloatList,
    max_parameters: FloatList,
    num_samples: int,
    device: Device,
) -> Tensor:
    _validate_equal_length_of_lists([min_parameters, max_parameters])
    num_dimensions = len(min_parameters)
    sobol_engine = torch.quasirandom.SobolEngine(num_dimensions)
    normalized_parameters = (
        sobol_engine.draw(num_samples).to(device).requires_grad_(True)
    )
    parameters = torch.tensor(min_parameters) + normalized_parameters * (
        torch.tensor(max_parameters) - torch.tensor(min_parameters)
    )
    if num_dimensions == 1:
        parameters = _reshape_parameters_for_one_dimension(parameters)
    return parameters


def _validate_equal_length_of_lists(lists: list[Lists]) -> None:
    list_lengths = [len(list_i) for list_i in lists]
    grouped_lengths = groupby(list_lengths)
    is_only_one_group = next(grouped_lengths, True) and not next(grouped_lengths, False)
    if not is_only_one_group:
        raise ParameterSamplingError(
            "It is expected that all list inputs are of equal length."
        )


def _reshape_parameters_for_one_dimension(parameters: Tensor) -> Tensor:
    return parameters.reshape((-1, 1))
