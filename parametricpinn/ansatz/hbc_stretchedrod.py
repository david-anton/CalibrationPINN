import torch

from parametricpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
)
from parametricpinn.types import Tensor


class HBCAnsatzStrategyStretchedRod:
    def __init__(self, displacement_left: float, range_coordinate: float) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._range_coordinate = range_coordinate

    def _boundary_data_func(self) -> float:
        return self._displacement_left

    def _distance_func(self, input_coor: Tensor) -> Tensor:
        return input_coor / self._range_coordinate

    def _extract_coordinates(self, input: Tensor) -> Tensor:
        if input.dim() == 1:
            return torch.unsqueeze(input[0], 0)
        return torch.unsqueeze(input[:, 0], 1)

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        x_coor = self._extract_coordinates(input)
        return self._boundary_data_func() + (
            self._distance_func(x_coor) * network(input)
        )


def create_standard_hbc_ansatz_stretched_rod(
    displacement_left: float, range_coordinate: float, network: StandardNetworks
) -> StandardAnsatz:
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(displacement_left, range_coordinate)
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_stretched_rod(
    displacement_left: float, range_coordinate: float, network: BayesianNetworks
) -> BayesianAnsatz:
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(displacement_left, range_coordinate)
    return BayesianAnsatz(network, ansatz_strategy)
