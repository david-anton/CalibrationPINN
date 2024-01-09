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
from parametricpinn.network.normalizednetwork import InputNormalizer, OutputRenormalizer
from parametricpinn.types import Device, Tensor

NetworkInputNormalizer: TypeAlias = InputNormalizer
AnsatzOutputNormalizer: TypeAlias = InputNormalizer
AnsatzOutputRenormalizer: TypeAlias = OutputRenormalizer


class NormalizedHBCAnsatzStrategyStretchedRod:
    def __init__(
        self,
        network_input_normalizer: NetworkInputNormalizer,
        ansatz_output_normalizer: AnsatzOutputNormalizer,
        distance_func: DistanceFunction,
        ansatz_output_renormalizer: AnsatzOutputRenormalizer,
        device: Device,
    ) -> None:
        super().__init__()
        self._displacement_left = torch.tensor([0.0], device=device)
        self._network_input_normalizer = network_input_normalizer
        self._ansatz_output_normalizer = ansatz_output_normalizer
        self._distance_func = distance_func
        self._ansatz_output_renormalizer = ansatz_output_renormalizer

    def _boundary_data_func(self) -> Tensor:
        return self._ansatz_output_normalizer(self._displacement_left)

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        coor = extract_coordinate_1d(input)
        norm_network_input = self._network_input_normalizer(input)
        norm_network_output = network(norm_network_input)
        norm_ansatz_output = self._boundary_data_func() + (
            self._distance_func(coor) * norm_network_output
        )
        return self._ansatz_output_renormalizer(norm_ansatz_output)


def create_standard_normalized_hbc_ansatz_stretched_rod(
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


def create_bayesian_normalized_hbc_ansatz_stretched_rod(
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
        network_input_normalizer=network_input_normalizer,
        ansatz_output_normalizer=ansatz_output_normalizer,
        distance_func=distance_func,
        ansatz_output_renormalizer=ansatz_output_renormalizer,
        device=device,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> NetworkInputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_ouput_normalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> AnsatzOutputNormalizer:
    return AnsatzOutputNormalizer(min_outputs, max_outputs)


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
) -> AnsatzOutputRenormalizer:
    return AnsatzOutputRenormalizer(min_outputs, max_outputs)
