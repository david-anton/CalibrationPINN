import torch

from parametricpinn.ansatz.ansatz_base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
)
from parametricpinn.types import Tensor


class HBCAnsatzStrategy2D:
    def __init__(
        self,
        displacement_x_right: float,
        displacement_y_bottom: float,
        range_coordinates: Tensor,
    ) -> None:
        super().__init__()
        self._boundary_data = torch.tensor(
            [displacement_x_right, displacement_y_bottom]
        ).to(range_coordinates.device)
        self._range_coordinates = range_coordinates

    def _boundary_data_func(self) -> Tensor:
        return self._boundary_data

    def _distance_func(self, x_coor: Tensor) -> Tensor:
        return x_coor / self._range_coordinates

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x[0:2]
        return x[:, 0:2]

    def __call__(self, x: Tensor, network: Networks) -> Tensor:
        x_coor = self._extract_coordinates(x)
        return self._boundary_data_func() + (self._distance_func(x_coor) * network(x))


def create_standard_hbc_ansatz_2D(
    displacement_x_right: float,
    displacement_y_bottom: float,
    range_coordinates: Tensor,
    network: StandardNetworks,
) -> StandardAnsatz:
    ansatz_strategy = HBCAnsatzStrategy2D(
        displacement_x_right, displacement_y_bottom, range_coordinates
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_2D(
    displacement_x_right: float,
    displacement_y_bottom: float,
    range_coordinates: Tensor,
    network: BayesianNetworks,
) -> BayesianAnsatz:
    ansatz_strategy = HBCAnsatzStrategy2D(
        displacement_x_right, displacement_y_bottom, range_coordinates
    )
    return BayesianAnsatz(network, ansatz_strategy)
