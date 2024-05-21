from typing import TypeAlias

import torch

from calibrationpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
    bind_output,
    extract_coordinates_2d,
    unbind_output,
)
from calibrationpinn.ansatz.distancefunctions import (
    DistanceFunction,
    distance_function_factory,
)
from calibrationpinn.network.normalizednetwork import (
    InputNormalizer,
    OutputRenormalizer,
)
from calibrationpinn.types import Device, Tensor

NetworkInputNormalizer: TypeAlias = InputNormalizer
AnsatzOutputNormalizer: TypeAlias = InputNormalizer
AnsatzOutputRenormalizer: TypeAlias = OutputRenormalizer


class NormalizedHBCAnsatzStrategyQuarterPlateWithHole:
    def __init__(
        self,
        network_input_normalizer: NetworkInputNormalizer,
        ansatz_output_normalizer_x: AnsatzOutputNormalizer,
        ansatz_output_normalizer_y: AnsatzOutputNormalizer,
        distance_func_x: DistanceFunction,
        distance_func_y: DistanceFunction,
        ansatz_output_renormalizer: AnsatzOutputRenormalizer,
        device: Device,
    ) -> None:
        super().__init__()
        self._boundary_data_x_right = torch.tensor([0.0], device=device)
        self._boundary_data_y_bottom = torch.tensor([0.0], device=device)
        self._network_input_normalizer = network_input_normalizer
        self._ansatz_output_normalizer_x = ansatz_output_normalizer_x
        self._ansatz_output_normalizer_y = ansatz_output_normalizer_y
        self._distance_func_x = distance_func_x
        self._distance_func_y = distance_func_y
        self._ansatz_output_renormalizer = ansatz_output_renormalizer

    def _boundary_data_func_x(self) -> Tensor:
        return self._ansatz_output_normalizer_x(self._boundary_data_x_right)

    def _boundary_data_func_y(self) -> Tensor:
        return self._ansatz_output_normalizer_y(self._boundary_data_y_bottom)

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        coor_x, coor_y = extract_coordinates_2d(input)
        norm_network_input = self._network_input_normalizer(input)
        norm_network_output = network(norm_network_input)
        norm_network_output_x, norm_network_output_y = unbind_output(
            norm_network_output
        )
        norm_ansatz_output_x = (
            self._boundary_data_func_x()
            + self._distance_func_x(coor_x) * norm_network_output_x
        )
        norm_ansatz_output_y = (
            self._boundary_data_func_y()
            + self._distance_func_y(coor_y) * norm_network_output_y
        )
        norm_ansatz_output = bind_output(norm_ansatz_output_x, norm_ansatz_output_y)
        return self._ansatz_output_renormalizer(norm_ansatz_output)


def create_standard_normalized_hbc_ansatz_quarter_plate_with_hole(
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
    device: Device,
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        min_inputs, max_inputs, min_outputs, max_outputs, distance_function_type, device
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_quarter_plate_with_hole(
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
    device: Device,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        min_inputs, max_inputs, min_outputs, max_outputs, distance_function_type, device
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    distance_func_type: str,
    device: Device,
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
        network_input_normalizer=network_input_normalizer,
        ansatz_output_normalizer_x=ansatz_output_normalizer_x,
        ansatz_output_normalizer_y=ansatz_output_normalizer_y,
        distance_func_x=distance_func_x,
        distance_func_y=distance_func_y,
        ansatz_output_renormalizer=ansatz_output_renormalizer,
        device=device,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> NetworkInputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_output_normalizers(
    min_outputs: Tensor, max_outputs: Tensor
) -> tuple[AnsatzOutputNormalizer, AnsatzOutputNormalizer]:
    output_normalizer_x = _create_one_ansatz_output_normalizer(
        min_outputs, max_outputs, index=0
    )
    output_normalizer_y = _create_one_ansatz_output_normalizer(
        min_outputs, max_outputs, index=1
    )
    return output_normalizer_x, output_normalizer_y


def _create_one_ansatz_output_normalizer(
    min_outputs: Tensor, max_outputs: Tensor, index: int
) -> AnsatzOutputNormalizer:
    return AnsatzOutputNormalizer(
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
) -> AnsatzOutputRenormalizer:
    return AnsatzOutputRenormalizer(min_outputs, max_outputs)
