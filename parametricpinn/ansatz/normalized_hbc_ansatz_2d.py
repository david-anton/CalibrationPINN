from typing import TypeAlias

import torch

from parametricpinn.ansatz.ansatz_base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
)
from parametricpinn.ansatz.hbc_ansatz_normalizers import (
    HBCAnsatzNormalizer,
    HBCAnsatzRenormalizer,
)
from parametricpinn.network.normalizednetwork import InputNormalizer
from parametricpinn.types import Tensor

NetworkInputNormalizer: TypeAlias = InputNormalizer


class NormalizedHBCAnsatzStrategy2D:
    def __init__(
        self,
        displacement_x_right: Tensor,
        displacement_y_bottom: Tensor,
        network_input_normalizer: NetworkInputNormalizer,
        hbc_ansatz_coordinate_normalizer: HBCAnsatzNormalizer,
        hbc_ansatz_output_normalizer: HBCAnsatzNormalizer,
        hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
    ) -> None:
        super().__init__()
        self._boundary_data = torch.tensor(
            [displacement_x_right, displacement_y_bottom]
        ).to(displacement_x_right.device)
        self._network_input_normalizer = network_input_normalizer
        self._hbc_ansatz_coordinate_normalizer = hbc_ansatz_coordinate_normalizer
        self._hbc_ansatz_output_normalizer = hbc_ansatz_output_normalizer
        self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

    def _boundary_data_func(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer(self._boundary_data)

    def _distance_func(self, x_coor: Tensor) -> Tensor:
        # It is assumed that the HBC is at the right (u_x=0) and bottom (u_y=0) edge.
        return self._hbc_ansatz_coordinate_normalizer(x_coor)

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x[0:2]
        return x[:, 0:2]

    def __call__(self, x: Tensor, network: Networks) -> Tensor:
        x_coor = self._extract_coordinates(x)
        norm_x = self._network_input_normalizer(x)
        norm_y = self._boundary_data_func() + (
            self._distance_func(x_coor) * network(norm_x)
        )
        return self._hbc_ansatz_output_renormalizer(norm_y)


def create_standard_normalized_hbc_ansatz_2D(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: StandardNetworks,
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_x_right,
        displacement_y_bottom,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_2D(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_x_right,
        displacement_y_bottom,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedHBCAnsatzStrategy2D:
    network_input_normalizer = _create_network_input_normalizer(min_inputs, max_inputs)
    ansatz_coordinate_normalizer = _create_ansatz_coordinate_normalizer(
        min_inputs, max_inputs
    )
    ansatz_output_normalizer = _create_ansatz_output_normalizer(
        min_outputs, max_outputs
    )
    ansatz_output_renormalizer = _create_ansatz_output_renormalizer(
        min_outputs, max_outputs
    )
    return NormalizedHBCAnsatzStrategy2D(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_coordinate_normalizer=ansatz_coordinate_normalizer,
        hbc_ansatz_output_normalizer=ansatz_output_normalizer,
        hbc_ansatz_output_renormalizer=ansatz_output_renormalizer,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> NetworkInputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_coordinate_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> HBCAnsatzNormalizer:
    idx_coordinate = slice(0, 2)
    return HBCAnsatzNormalizer(min_inputs[idx_coordinate], max_inputs[idx_coordinate])


def _create_ansatz_output_normalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzNormalizer:
    return HBCAnsatzNormalizer(min_outputs, max_outputs)


def _create_ansatz_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzRenormalizer:
    return HBCAnsatzRenormalizer(min_outputs, max_outputs)
