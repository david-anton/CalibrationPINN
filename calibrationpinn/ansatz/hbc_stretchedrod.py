import torch

from calibrationpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
    extract_coordinate_1d,
)
from calibrationpinn.ansatz.distancefunctions import (
    DistanceFunction,
    distance_function_factory,
)
from calibrationpinn.types import Device, Tensor


class HBCAnsatzStrategyStretchedRod:
    def __init__(self, distance_func: DistanceFunction, device: Device) -> None:
        super().__init__()
        self._displacement_left = torch.tensor([0.0], device=device)
        self._distance_func = distance_func

    def _boundary_data_func(self) -> Tensor:
        return self._displacement_left

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor = extract_coordinate_1d(input)
        return self._boundary_data_func() + (
            self._distance_func(input_coor) * network(input)
        )


def create_standard_hbc_ansatz_stretched_rod(
    range_coordinate: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
    device: Device,
) -> StandardAnsatz:
    distance_func = _create_distance_function(distance_function_type, range_coordinate)
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(distance_func, device)
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_stretched_rod(
    range_coordinate: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
    device: Device,
) -> BayesianAnsatz:
    distance_func = _create_distance_function(distance_function_type, range_coordinate)
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(distance_func, device)
    return BayesianAnsatz(network, ansatz_strategy)


def _create_distance_function(
    distance_func_type: str, range_coordinate: Tensor
) -> DistanceFunction:
    device = range_coordinate.device
    boundary_coordinate = torch.tensor([0.0], device=device)
    return distance_function_factory(
        distance_func_type, range_coordinate, boundary_coordinate
    )
