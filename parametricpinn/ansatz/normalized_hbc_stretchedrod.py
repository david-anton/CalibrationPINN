from typing import TypeAlias

import torch

from parametricpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
    extract_coordinate_1d,
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


class NormalizedHBCAnsatzStrategyStretchedRod:
    def __init__(
        self,
        displacement_left: Tensor,
        network_input_normalizer: NetworkInputNormalizer,
        hbc_ansatz_output_normalizer: HBCAnsatzNormalizer,
        distance_func: DistanceFunction,
        hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
    ) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._network_input_normalizer = network_input_normalizer
        self._hbc_ansatz_output_normalizer = hbc_ansatz_output_normalizer
        self._distance_func = distance_func
        self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

    def _boundary_data_func(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer(self._displacement_left)

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor = extract_coordinate_1d(input)
        norm_input = self._network_input_normalizer(input)
        norm_output = self._boundary_data_func() + (
            self._distance_func(input_coor) * network(norm_input)
        )
        return self._hbc_ansatz_output_renormalizer(norm_output)


def create_standard_normalized_hbc_ansatz_stretched_rod(
    displacement_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_stretched_rod(
    displacement_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    displacement_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    distance_func_type: str,
) -> NormalizedHBCAnsatzStrategyStretchedRod:
    network_input_normalizer = _create_network_input_normalizer(min_inputs, max_inputs)
    ansatz_output_normalizer = _create_ansatz_ouput_normalizer(min_outputs, max_outputs)
    distance_func = _create_distance_function(
        distance_func_type, min_inputs, max_inputs
    )
    ansatz_output_renormalizer = _create_ansatz_output_renormalizer(
        min_outputs, max_outputs
    )
    return NormalizedHBCAnsatzStrategyStretchedRod(
        displacement_left=displacement_left,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_output_normalizer=ansatz_output_normalizer,
        distance_func=distance_func,
        hbc_ansatz_output_renormalizer=ansatz_output_renormalizer,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> InputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_ouput_normalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzNormalizer:
    return HBCAnsatzNormalizer(min_outputs, max_outputs)


def _create_distance_function(
    distance_func_type: str, min_inputs: Tensor, max_inputs: Tensor
) -> DistanceFunction:
    range_coordinate = torch.unsqueeze(max_inputs[0] - min_inputs[0], dim=0)
    device = range_coordinate.device
    boundary_coordinate = torch.tensor([0.0], device=device)
    return distance_function_factory(
        distance_func_type, range_coordinate, boundary_coordinate
    )


def _create_ansatz_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzRenormalizer:
    return HBCAnsatzRenormalizer(min_outputs, max_outputs)
