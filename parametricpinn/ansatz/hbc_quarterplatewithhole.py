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
from parametricpinn.ansatz.distancefunctions import (
    DistanceFunction,
    distance_function_factory,
)
from parametricpinn.types import Tensor


class HBCAnsatzStrategyQuarterPlateWithHole:
    def __init__(
        self,
        displacement_x_right: Tensor,
        displacement_y_bottom: Tensor,
        distance_func_x: DistanceFunction,
        distamce_func_y: DistanceFunction,
    ) -> None:
        super().__init__()
        self._boundary_data_x_right = displacement_x_right
        self._boundary_data_y_bottom = displacement_y_bottom
        self._distance_func_x = distance_func_x
        self._distance_func_y = distamce_func_y

    def _boundary_data_func_x(self) -> Tensor:
        return self._boundary_data_x_right

    def _boundary_data_func_y(self) -> Tensor:
        return self._boundary_data_y_bottom

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
    distance_function_type: str,
) -> StandardAnsatz:
    distance_func_x, distance_func_y = _create_distance_functions(
        distance_function_type, range_coordinates
    )
    ansatz_strategy = HBCAnsatzStrategyQuarterPlateWithHole(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        distance_func_x=distance_func_x,
        distamce_func_y=distance_func_y,
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_quarter_plate_with_hole(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    range_coordinates: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
) -> BayesianAnsatz:
    distance_func_x, distance_func_y = _create_distance_functions(
        distance_function_type, range_coordinates
    )
    ansatz_strategy = HBCAnsatzStrategyQuarterPlateWithHole(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        distance_func_x=distance_func_x,
        distamce_func_y=distance_func_y,
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_distance_functions(
    distance_func_type: str, range_coordinates: Tensor
) -> tuple[DistanceFunction, DistanceFunction]:
    range_coordinate_x = torch.unsqueeze(range_coordinates[0], dim=0)
    range_coordinate_y = torch.unsqueeze(range_coordinates[1], dim=0)
    distance_func_x = _create_distance_one_function(
        distance_func_type, range_coordinate_x
    )
    distance_func_y = _create_distance_one_function(
        distance_func_type, range_coordinate_y
    )
    return distance_func_x, distance_func_y


def _create_distance_one_function(
    distance_func_type: str, range_coordinate: Tensor
) -> DistanceFunction:
    return distance_function_factory(distance_func_type, range_coordinate)
