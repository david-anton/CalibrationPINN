from typing import Callable, TypeAlias

import torch

from parametricpinn.errors import DistanceFunctionConfigError
from parametricpinn.types import Tensor

DistanceFunction: TypeAlias = Callable[[Tensor], Tensor]


def normalized_linear_distance_function_factory(
    range_coordinate: Tensor,
) -> DistanceFunction:
    def distance_func(input_coord: Tensor) -> Tensor:
        return input_coord / range_coordinate

    return distance_func


def sigmoid_distance_function_factory() -> DistanceFunction:
    def distance_func(input_coor: Tensor) -> Tensor:
        return (2 * torch.exp(input_coor) / torch.exp(input_coor) + 1) - 1

    return distance_func


def distance_function_factory(
    type_str: str, range_coordinate: Tensor
) -> DistanceFunction:
    if type_str == "normalized_linear":
        return normalized_linear_distance_function_factory(range_coordinate)
    elif type_str == "sigmoid":
        return sigmoid_distance_function_factory()
    else:
        raise DistanceFunctionConfigError(
            f"There is no implementation for the requested distance function {type_str}."
        )
