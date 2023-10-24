import torch

from parametricpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
    extract_coordinate_1d,
)
from parametricpinn.types import Tensor


class HBCAnsatzStrategyStretchedRod:
    def __init__(self, displacement_left: Tensor, range_coordinate: Tensor) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._range_coordinate = range_coordinate

    def _boundary_data_func(self) -> Tensor:
        return self._displacement_left

    def _distance_func(self, input_coor: Tensor) -> Tensor:
        return input_coor / self._range_coordinate

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor = extract_coordinate_1d(input)
        return self._boundary_data_func() + (
            self._distance_func(input_coor) * network(input)
        )


def create_standard_hbc_ansatz_stretched_rod(
    displacement_left: Tensor, range_coordinate: Tensor, network: StandardNetworks
) -> StandardAnsatz:
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(displacement_left, range_coordinate)
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_stretched_rod(
    displacement_left: Tensor, range_coordinate: Tensor, network: BayesianNetworks
) -> BayesianAnsatz:
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(displacement_left, range_coordinate)
    return BayesianAnsatz(network, ansatz_strategy)
