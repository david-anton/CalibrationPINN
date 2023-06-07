from typing import TypeAlias

import torch
import torch.nn as nn

from parametricpinn.ansatz.hbc_ansatz_normalizers import (
    HBCAnsatzNormalizer,
    HBCAnsatzRenormalizer,
)
from parametricpinn.network.normalizednetwork import InputNormalizer
from parametricpinn.types import Module, Tensor

NetworkInputNormalizer: TypeAlias = InputNormalizer


class NormalizedHBCAnsatz2D(nn.Module):
    def __init__(
        self,
        displacement_x_right: Tensor,
        displacement_y_bottom: Tensor,
        network: Module,
        network_input_normalizer: NetworkInputNormalizer,
        hbc_ansatz_coordinate_normalizer: HBCAnsatzNormalizer,
        hbc_ansatz_output_normalizer: HBCAnsatzNormalizer,
        hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
    ) -> None:
        super().__init__()
        self._boundary_data = torch.tensor(
            [displacement_x_right, displacement_y_bottom]
        )
        self._network = network
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

    def forward(self, x: Tensor) -> Tensor:
        x_coor = self._extract_coordinates(x)
        norm_x = self._network_input_normalizer(x)
        norm_y = self._boundary_data_func() + (
            self._distance_func(x_coor) * self._network(norm_x)
        )
        return self._hbc_ansatz_output_renormalizer(norm_y)


def create_normalized_hbc_ansatz_2D(
    displacement_x_right: Tensor,
    displacement_y_bottom: Tensor,
    network: Module,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedHBCAnsatz2D:
    network_input_normalizer = NetworkInputNormalizer(
        min_inputs=min_inputs, max_inputs=max_inputs
    )
    idx_coordinate = slice(0, 2)
    hbc_ansatz_coordinate_normalizer = HBCAnsatzNormalizer(
        min_values=min_inputs[idx_coordinate],
        max_values=max_inputs[idx_coordinate],
    )
    hbc_ansatz_output_normalizer = HBCAnsatzNormalizer(
        min_values=min_outputs, max_values=max_outputs
    )
    hbc_ansatz_output_renormalizer = HBCAnsatzRenormalizer(
        min_values=min_outputs, max_values=max_outputs
    )
    return NormalizedHBCAnsatz2D(
        displacement_x_right=displacement_x_right,
        displacement_y_bottom=displacement_y_bottom,
        network=network,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_coordinate_normalizer=hbc_ansatz_coordinate_normalizer,
        hbc_ansatz_output_normalizer=hbc_ansatz_output_normalizer,
        hbc_ansatz_output_renormalizer=hbc_ansatz_output_renormalizer,
    )
