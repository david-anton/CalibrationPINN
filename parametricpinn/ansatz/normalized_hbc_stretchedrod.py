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
        hbc_ansatz_coordinate_normalizer: HBCAnsatzNormalizer,
        hbc_ansatz_output_normalizer: HBCAnsatzNormalizer,
        hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
    ) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._network_input_normalizer = network_input_normalizer
        self._hbc_ansatz_coordinate_normalizer = hbc_ansatz_coordinate_normalizer
        self._hbc_ansatz_output_normalizer = hbc_ansatz_output_normalizer
        self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

    def _boundary_data_func(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer(self._displacement_left)

    def _distance_func(self, input_coor: Tensor) -> Tensor:
        # It is assumed that the HBC is at x_coor=0.
        return self._hbc_ansatz_coordinate_normalizer(input_coor)

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
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_left, min_inputs, max_inputs, min_outputs, max_outputs
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_stretched_rod(
    displacement_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_left, min_inputs, max_inputs, min_outputs, max_outputs
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    displacement_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedHBCAnsatzStrategyStretchedRod:
    network_input_normalizer = _create_network_input_normalizer(min_inputs, max_inputs)
    ansatz_coordinate_normalizer = _create_ansatz_coordinate_normalizer(
        min_inputs, max_inputs
    )
    ansatz_output_normalizer = _create_ansatz_ouput_normalizer(min_outputs, max_outputs)
    ansatz_output_renormalizer = _create_ansatz_output_renormalizer(
        min_outputs, max_outputs
    )
    return NormalizedHBCAnsatzStrategyStretchedRod(
        displacement_left=displacement_left,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_coordinate_normalizer=ansatz_coordinate_normalizer,
        hbc_ansatz_output_normalizer=ansatz_output_normalizer,
        hbc_ansatz_output_renormalizer=ansatz_output_renormalizer,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> InputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_coordinate_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> HBCAnsatzNormalizer:
    idx_coordinate = 0
    return HBCAnsatzNormalizer(min_inputs[idx_coordinate], max_inputs[idx_coordinate])


def _create_ansatz_ouput_normalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzNormalizer:
    return HBCAnsatzNormalizer(min_outputs, max_outputs)


def _create_ansatz_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzRenormalizer:
    return HBCAnsatzRenormalizer(min_outputs, max_outputs)
