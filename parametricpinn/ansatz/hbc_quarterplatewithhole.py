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
from parametricpinn.types import Device, Tensor


class HBCAnsatzStrategyQuarterPlateWithHole:
    def __init__(
        self,
        distance_func_x: DistanceFunction,
        distamce_func_y: DistanceFunction,
        device: Device
    ) -> None:
        super().__init__()
        self._boundary_data_x_right = torch.tensor([0.0], device=device)
        self._boundary_data_y_bottom = torch.tensor([0.0], device=device)
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
    range_coordinates: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
    device: Device
) -> StandardAnsatz:
    distance_func_x, distance_func_y = _create_distance_functions(
        distance_function_type, range_coordinates
    )
    ansatz_strategy = HBCAnsatzStrategyQuarterPlateWithHole(
        distance_func_x=distance_func_x,
        distamce_func_y=distance_func_y,
        device=device
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_quarter_plate_with_hole(
    range_coordinates: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
    device: Device
) -> BayesianAnsatz:
    distance_func_x, distance_func_y = _create_distance_functions(
        distance_function_type, range_coordinates
    )
    ansatz_strategy = HBCAnsatzStrategyQuarterPlateWithHole(
        distance_func_x=distance_func_x,
        distamce_func_y=distance_func_y,
        device=device
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_distance_functions(
    distance_func_type: str, range_coordinates: Tensor
) -> tuple[DistanceFunction, DistanceFunction]:
    range_coordinate_x = torch.unsqueeze(range_coordinates[0], dim=0)
    range_coordinate_y = torch.unsqueeze(range_coordinates[1], dim=0)
    distance_func_x = _create_one_distance_function(
        distance_func_type, range_coordinate_x
    )
    distance_func_y = _create_one_distance_function(
        distance_func_type, range_coordinate_y
    )
    return distance_func_x, distance_func_y


def _create_one_distance_function(
    distance_func_type: str, range_coordinate: Tensor
) -> DistanceFunction:
    device = range_coordinate.device
    boundary_coordinate = torch.tensor([0.0], device=device)
    return distance_function_factory(
        distance_func_type, range_coordinate, boundary_coordinate
    )
