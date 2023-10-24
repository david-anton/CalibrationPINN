import torch

from parametricpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
    bind_output,
    extract_coordinates_2d,
    unbind_output,
)
from parametricpinn.types import Tensor


class HBCAnsatzStrategyQuarterPlateWithHole:
    def __init__(
        self,
        displacement_x_right: Tensor,
        displacement_y_bottom: Tensor,
        range_coordinates: Tensor,
    ) -> None:
        super().__init__()
        self._boundary_data_x_right = displacement_x_right
        self._boundary_data_y_bottom = displacement_y_bottom
        self._range_coordinates_x = torch.unsqueeze(range_coordinates[0], dim=0)
        self._range_coordinates_y = torch.unsqueeze(range_coordinates[1], dim=0)

    def _boundary_data_func_x(self) -> Tensor:
        return self._boundary_data_x_right

    def _boundary_data_func_y(self) -> Tensor:
        return self._boundary_data_y_bottom

    def _distance_func_x(self, input_coor_x: Tensor) -> Tensor:
        # Boundary condition u_x=0 is at x=0
        return input_coor_x / self._range_coordinates_x

    def _distance_func_y(self, input_coor_y: Tensor) -> Tensor:
        # Boundary condition u_y=0 is at y=0
        return input_coor_y / self._range_coordinates_y

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor_x, input_coor_y = extract_coordinates_2d(input)
        network_output = network(input)
        network_output_x, network_output_y = unbind_output(network_output)
        output_x = (
            self._boundary_data_func_x()
            + self._distance_func_x(input_coor_x) * network_output_x
        )
        output_y = (
            self._boundary_data_func_y()
            + self._distance_func_y(input_coor_y) * network_output_y
        )
        return bind_output(output_x, output_y)


def create_standard_hbc_ansatz_quarter_plate_with_hole(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    range_coordinates: Tensor,
    network: StandardNetworks,
) -> StandardAnsatz:
    ansatz_strategy = HBCAnsatzStrategyQuarterPlateWithHole(
        displacement_x_right, displacement_y_bottom, range_coordinates
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_quarter_plate_with_hole(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    range_coordinates: Tensor,
    network: BayesianNetworks,
) -> BayesianAnsatz:
    ansatz_strategy = HBCAnsatzStrategyQuarterPlateWithHole(
        displacement_x_right, displacement_y_bottom, range_coordinates
    )
    return BayesianAnsatz(network, ansatz_strategy)
