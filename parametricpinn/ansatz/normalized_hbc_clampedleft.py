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


class NormalizedHBCAnsatzStrategyClampedLeft:
    def __init__(
        self,
        displacement_x_left: Tensor,
        network_input_normalizer: NetworkInputNormalizer,
        hbc_ansatz_output_normalizer_x: HBCAnsatzNormalizer,
        distance_func_x: DistanceFunction,
        hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
    ) -> None:
        super().__init__()
        self._boundary_data_x_left = displacement_x_left
        self._network_input_normalizer = network_input_normalizer
        self._hbc_ansatz_output_normalizer_x = hbc_ansatz_output_normalizer_x
        self._distance_func_x = distance_func_x
        self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

    def _boundary_data_func_x(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer_x(self._boundary_data_x_left)

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor_x, _ = extract_coordinates_2d(input)
        norm_input = self._network_input_normalizer(input)
        network_output = network(norm_input)
        network_output_x, network_output_y = unbind_output(network_output)
        norm_output_x = (
            self._boundary_data_func_x()
            + self._distance_func_x(input_coor_x) * network_output_x
        )
        norm_output_y = network_output_y
        norm_output = bind_output(norm_output_x, norm_output_y)
        return self._hbc_ansatz_output_renormalizer(norm_output)


def create_standard_normalized_hbc_ansatz_clamped_left(
    coordinate_x_left: Tensor,
    displacement_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        coordinate_x_left,
        displacement_x_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_clamped_left(
    coordinate_x_left: Tensor,
    displacement_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        coordinate_x_left,
        displacement_x_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    coordinate_x_left: Tensor,
    displacement_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    distance_func_type: str,
) -> NormalizedHBCAnsatzStrategyClampedLeft:
    network_input_normalizer = _create_network_input_normalizer(min_inputs, max_inputs)
    ansatz_output_normalizer_x = _create_ansatz_output_normalizer_x(
        min_outputs, max_outputs
    )
    distance_func_x = _create_distance_function_x(
        distance_func_type, min_inputs, max_inputs, coordinate_x_left
    )
    ansatz_output_renormalizer = _create_ansatz_output_renormalizer(
        min_outputs, max_outputs
    )
    return NormalizedHBCAnsatzStrategyClampedLeft(
        displacement_x_left=displacement_x_left,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_output_normalizer_x=ansatz_output_normalizer_x,
        distance_func_x=distance_func_x,
        hbc_ansatz_output_renormalizer=ansatz_output_renormalizer,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> NetworkInputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_output_normalizer_x(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzNormalizer:
    idx_output = 0
    return HBCAnsatzNormalizer(
        torch.unsqueeze(min_outputs[idx_output], dim=0),
        torch.unsqueeze(max_outputs[idx_output], dim=0),
    )


def _create_distance_function_x(
    distance_func_type: str,
    min_inputs: Tensor,
    max_inputs: Tensor,
    coordinate_x_left: Tensor,
) -> DistanceFunction:
    idx_coordinate = 0
    range_coordinate_x = torch.unsqueeze(
        max_inputs[idx_coordinate] - min_inputs[idx_coordinate], dim=0
    )
    device = range_coordinate_x.device
    boundary_coordinate_x = coordinate_x_left.to(device)
    return distance_function_factory(
        distance_func_type, range_coordinate_x, boundary_coordinate_x
    )


def _create_ansatz_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzRenormalizer:
    return HBCAnsatzRenormalizer(min_outputs, max_outputs)
