import torch

from parametricpinn.ansatz.ansatz_base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
)
from parametricpinn.types import Tensor


class HBCAnsatzStrategy1D:
    def __init__(self, displacement_left: float, range_coordinate: float) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._range_coordinate = range_coordinate

    def _boundary_data_func(self) -> float:
        return self._displacement_left

    def _distance_func(self, x_coor: Tensor) -> Tensor:
        return x_coor / self._range_coordinate

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return torch.unsqueeze(x[0], 0)
        return torch.unsqueeze(x[:, 0], 1)

    def __call__(self, x: Tensor, network: Networks) -> Tensor:
        x_coor = self._extract_coordinates(x)
        return self._boundary_data_func() + (self._distance_func(x_coor) * network(x))


def create_standard_hbc_ansatz_1D(
    displacement_left: float, range_coordinate: float, network: StandardNetworks
) -> StandardAnsatz:
    ansatz_strategy = HBCAnsatzStrategy1D(displacement_left, range_coordinate)
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_1D(
    displacement_left: float, range_coordinate: float, network: BayesianNetworks
) -> BayesianAnsatz:
    ansatz_strategy = HBCAnsatzStrategy1D(displacement_left, range_coordinate)
    return BayesianAnsatz(network, ansatz_strategy)
