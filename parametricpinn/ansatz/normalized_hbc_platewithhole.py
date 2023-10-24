from typing import TypeAlias

import torch

from parametricpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
)
from parametricpinn.ansatz.hbc_normalizers import (
    HBCAnsatzNormalizer,
    HBCAnsatzRenormalizer,
)
from parametricpinn.network.normalizednetwork import InputNormalizer
from parametricpinn.types import Tensor

NetworkInputNormalizer: TypeAlias = InputNormalizer


class NormalizedHBCAnsatzStrategyPlateWithHole:
    def __init__(
        self,
        displacement_x_left: Tensor,
        network_input_normalizer: NetworkInputNormalizer,
        hbc_ansatz_coordinate_normalizer_x: HBCAnsatzNormalizer,
        hbc_ansatz_output_normalizer_x: HBCAnsatzNormalizer,
        hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
    ) -> None:
        super().__init__()
        self._boundary_data_x_left = displacement_x_left
        self._network_input_normalizer = network_input_normalizer
        self._hbc_ansatz_coordinate_normalizer_x = hbc_ansatz_coordinate_normalizer_x
        self._hbc_ansatz_output_normalizer_x = hbc_ansatz_output_normalizer_x
        self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

    def _boundary_data_func_x(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer_x(self._boundary_data_x_left)

    def _distance_func_x(self, input_coor_x: Tensor) -> Tensor:
        # Boundary condition u_x=0 is at x=0
        return self._hbc_ansatz_coordinate_normalizer_x(input_coor_x)

    def _extract_coordinates_x(self, input: Tensor) -> Tensor:
        if input.dim() == 1:
            return input[0]
        return input[:, 0].reshape((-1, 1))

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor_x = self._extract_coordinates_x(input)
        norm_input = self._network_input_normalizer(input)
        network_output = network(norm_input)
        network_output_x, network_output_y = torch.unbind(network_output, dim=1)
        network_output_x = network_output_x.reshape((-1, 1))
        network_output_y = network_output_y.reshape((-1, 1))
        norm_output_x = (
            self._boundary_data_func_x()
            + self._distance_func_x(input_coor_x) * network_output_x
        )
        norm_output_y = network_output_y
        norm_output = torch.concat((norm_output_x, norm_output_y), dim=1)
        return self._hbc_ansatz_output_renormalizer(norm_output)


def create_standard_normalized_hbc_ansatz_plate_with_hole(
    displacement_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: StandardNetworks,
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_x_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_plate_with_hole(
    displacement_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        displacement_x_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    displacement_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedHBCAnsatzStrategyPlateWithHole:
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
    return NormalizedHBCAnsatzStrategyPlateWithHole(
        displacement_x_left=displacement_x_left,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_coordinate_normalizer_x=ansatz_coordinate_normalizer,
        hbc_ansatz_output_normalizer_x=ansatz_output_normalizer,
        hbc_ansatz_output_renormalizer=ansatz_output_renormalizer,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> NetworkInputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_ansatz_coordinate_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> HBCAnsatzNormalizer:
    idx_coordinates = slice(0, 2)
    return HBCAnsatzNormalizer(min_inputs[idx_coordinates], max_inputs[idx_coordinates])


def _create_ansatz_output_normalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzNormalizer:
    return HBCAnsatzNormalizer(min_outputs, max_outputs)


def _create_ansatz_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> HBCAnsatzRenormalizer:
    return HBCAnsatzRenormalizer(min_outputs, max_outputs)
