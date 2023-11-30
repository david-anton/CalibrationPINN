from typing import TypeAlias

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
from parametricpinn.ansatz.hbc_normalizers import (
    HBCAnsatzNormalizer,
    HBCAnsatzRenormalizer,
)
from parametricpinn.network.normalizednetwork import InputNormalizer
from parametricpinn.types import Tensor

NetworkInputNormalizer: TypeAlias = InputNormalizer


class NormalizedHBCAnsatzStrategyQuarterPlateWithHole:
    def __init__(
        self,
        displacement_x_right: Tensor,
        displacement_y_bottom: Tensor,
        network_input_normalizer: NetworkInputNormalizer,
        hbc_ansatz_output_normalizer_x: HBCAnsatzNormalizer,
        hbc_ansatz_output_normalizer_y: HBCAnsatzNormalizer,
        distance_func_x: DistanceFunction,
        distance_func_y: DistanceFunction,
        hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
    ) -> None:
        super().__init__()
        self._boundary_data_x_right = displacement_x_right
        self._boundary_data_y_bottom = displacement_y_bottom
        self._network_input_normalizer = network_input_normalizer
        self._hbc_ansatz_output_normalizer_x = hbc_ansatz_output_normalizer_x
        self._hbc_ansatz_output_normalizer_y = hbc_ansatz_output_normalizer_y
        self._distance_func_x = distance_func_x
        self._distance_func_y = distance_func_y
        self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

    def _boundary_data_func_x(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer_x(self._boundary_data_x_right)

    def _boundary_data_func_y(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer_y(self._boundary_data_y_bottom)

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor_x, input_coor_y = extract_coordinates_2d(input)
        norm_input = self._network_input_normalizer(input)
        norm_network_output = network(norm_input)
        norm_network_output_x, norm_network_output_y = unbind_output(
            norm_network_output
        )
        norm_output_x = (
            self._boundary_data_func_x()
            + self._distance_func_x(input_coor_x) * norm_network_output_x
        )
        norm_output_y = (
            self._boundary_data_func_y()
            + self._distance_func_y(input_coor_y) * norm_network_output_y
        )
        norm_output = bind_output(norm_output_x, norm_output_y)
        return self._hbc_ansatz_output_renormalizer(norm_output)


def create_standard_normalized_hbc_ansatz_quarter_plate_with_hole(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_x_right,
        displacement_y_bottom,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_quarter_plate_with_hole(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_x_right,
        displacement_y_bottom,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    distance_func_type: str,
) -> NormalizedHBCAnsatzStrategyQuarterPlateWithHole:
    network_input_normalizer = _create_network_input_normalizer(min_inputs, max_inputs)
    (
        ansatz_output_normalizer_x,
        ansatz_output_normalizer_y,
    ) = _create_ansatz_output_normalizers(min_outputs, max_outputs)
    distance_func_x, distance_func_y = _create_distance_functions(
        distance_func_type, min_inputs, max_inputs
    )
    ansatz_output_renormalizer = _create_ansatz_output_renormalizer(
        min_outputs, max_outputs
    )
    return NormalizedHBCAnsatzStrategyQuarterPlateWithHole(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_output_normalizer_x=ansatz_output_normalizer_x,
        hbc_ansatz_output_normalizer_y=ansatz_output_normalizer_y,
        distance_func_x=distance_func_x,
        distance_func_y=distance_func_y,
        hbc_ansatz_output_renormalizer=ansatz_output_renormalizer,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> NetworkInputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_output_normalizers(
    min_outputs: Tensor, max_outputs: Tensor
) -> tuple[HBCAnsatzNormalizer, HBCAnsatzNormalizer]:
    output_normalizer_x = _create_one_ansatz_output_normalizer(
        min_outputs, max_outputs, index=0
    )
    output_normalizer_y = _create_one_ansatz_output_normalizer(
        min_outputs, max_outputs, index=1
    )
    return output_normalizer_x, output_normalizer_y


def _create_one_ansatz_output_normalizer(
    min_outputs: Tensor, max_outputs: Tensor, index: int
) -> HBCAnsatzNormalizer:
    return HBCAnsatzNormalizer(
        torch.unsqueeze(min_outputs[index], dim=0),
        torch.unsqueeze(max_outputs[index], dim=0),
    )


def _create_distance_functions(
    distance_func_type: str, min_inputs: Tensor, max_inputs: Tensor
) -> tuple[DistanceFunction, DistanceFunction]:
    distance_func_x = _create_one_distance_function(
        distance_func_type, min_inputs, max_inputs, index=0
    )
    distance_func_y = _create_one_distance_function(
        distance_func_type, min_inputs, max_inputs, index=1
    )
    return distance_func_x, distance_func_y


def _create_one_distance_function(
    distance_func_type: str, min_inputs: Tensor, max_inputs: Tensor, index: int
) -> DistanceFunction:
    range_coordinate = torch.unsqueeze(max_inputs[index] - min_inputs[index], dim=0)
    device = range_coordinate.device
    boundary_coordinate = torch.tensor([0.0], device=device)
    return distance_function_factory(
        distance_func_type, range_coordinate, boundary_coordinate
    )


def _create_ansatz_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzRenormalizer:
    return HBCAnsatzRenormalizer(min_outputs, max_outputs)
