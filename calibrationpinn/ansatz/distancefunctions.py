from typing import Callable, TypeAlias

import torch

from calibrationpinn.errors import DistanceFunctionConfigError
from calibrationpinn.types import Tensor

DistanceFunction: TypeAlias = Callable[[Tensor], Tensor]


def normalized_linear_distance_function_factory(
    range_coordinate: Tensor,
    boundary_coordinate: Tensor,
) -> DistanceFunction:
    def distance_func(input_coordinate: Tensor) -> Tensor:
        relative_coordinate = input_coordinate - boundary_coordinate
        return (relative_coordinate) / range_coordinate

    return distance_func


def sigmoid_distance_function_factory(boundary_coordinate: Tensor) -> DistanceFunction:
    def distance_func(input_coordinate: Tensor) -> Tensor:
        relative_coordinate = input_coordinate - boundary_coordinate
        return (
            (2 * torch.exp(relative_coordinate)) / (torch.exp(relative_coordinate) + 1)
        ) - 1

    return distance_func


def distance_function_factory(
    type_str: str,
    range_coordinate: Tensor,
    boundary_coordinate: Tensor,
) -> DistanceFunction:
    if type_str == "normalized linear":
        return normalized_linear_distance_function_factory(
            range_coordinate, boundary_coordinate
        )
    elif type_str == "sigmoid":
        return sigmoid_distance_function_factory(boundary_coordinate)
    else:
        raise DistanceFunctionConfigError(
            f"There is no implementation for the requested distance function {type_str}."
        )
