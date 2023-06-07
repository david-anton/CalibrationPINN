import torch
import torch.nn as nn

from parametricpinn.types import Module, Tensor


class NetworkInputNormalizer(nn.Module):
    def __init__(self, min_inputs: Tensor, max_inputs: Tensor) -> None:
        super().__init__()
        self._min_inputs = min_inputs
        self._max_inputs = max_inputs
        self._input_ranges = max_inputs - min_inputs
        self._atol = torch.tensor([1e-7]).to(self._input_ranges.device)

    def forward(self, x: Tensor) -> Tensor:
        return (
            (
                ((x - self._min_inputs) + self._atol)
                / (self._input_ranges + 2 * self._atol)
            )
            * 2.0
        ) - 1.0


class HBCAnsatzCoordinateNormalizer(nn.Module):
    def __init__(self, min_coordinate: Tensor, max_coordinate: Tensor) -> None:
        super().__init__()
        self._coordinate_range = max_coordinate - min_coordinate

    def forward(self, x: Tensor) -> Tensor:
        return x / self._coordinate_range


class HBCAnsatzOutputNormalizer(nn.Module):
    def __init__(self, min_outputs: Tensor, max_outputs: Tensor) -> None:
        super().__init__()
        self._output_ranges = max_outputs - min_outputs

    def forward(self, x: Tensor) -> Tensor:
        return x / self._output_ranges


class HBCAnsatzOutputRenormalizer(nn.Module):
    def __init__(self, min_outputs: Tensor, max_outputs: Tensor) -> None:
        super().__init__()
        self._output_ranges = max_outputs - min_outputs

    def forward(self, x: Tensor) -> Tensor:
        return x * self._output_ranges


class NormalizedHBCAnsatz1D(nn.Module):
    def __init__(
        self,
        displacement_left: Tensor,
        network: Module,
        network_input_normalizer: NetworkInputNormalizer,
        hbc_ansatz_coordinate_normalizer: HBCAnsatzCoordinateNormalizer,
        hbc_ansatz_output_normalizer: HBCAnsatzOutputNormalizer,
        hbc_ansatz_output_renormalizer: HBCAnsatzOutputRenormalizer,
    ) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._network = network
        self._network_input_normalizer = network_input_normalizer
        self._hbc_ansatz_coordinate_normalizer = hbc_ansatz_coordinate_normalizer
        self._hbc_ansatz_output_normalizer = hbc_ansatz_output_normalizer
        self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

    def _boundary_data(self) -> Tensor:
        return self._hbc_ansatz_output_normalizer(self._displacement_left)

    def _distance_function(self, x_coor: Tensor) -> Tensor:
        # It is assumed that the HBC is at x_coor=0.
        return self._hbc_ansatz_coordinate_normalizer(x_coor)

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return torch.unsqueeze(x[0], 0)
        return torch.unsqueeze(x[:, 0], 1)

    def forward(self, x: Tensor) -> Tensor:
        x_coor = self._extract_coordinates(x)
        norm_x = self._network_input_normalizer(x)
        norm_y = self._boundary_data() + (
            self._distance_function(x_coor) * self._network(norm_x)
        )
        return self._hbc_ansatz_output_renormalizer(norm_y)


def create_normalized_hbc_ansatz_1D(
    displacement_left: Tensor,
    network: Module,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedHBCAnsatz1D:
    network_input_normalizer = NetworkInputNormalizer(
        min_inputs=min_inputs, max_inputs=max_inputs
    )
    idx_coordinate = 0
    hbc_ansatz_coordinate_normalizer = HBCAnsatzCoordinateNormalizer(
        min_coordinate=min_inputs[idx_coordinate],
        max_coordinate=max_inputs[idx_coordinate],
    )
    hbc_ansatz_output_normalizer = HBCAnsatzOutputNormalizer(
        min_outputs=min_outputs, max_outputs=max_outputs
    )
    hbc_ansatz_output_renormalizer = HBCAnsatzOutputRenormalizer(
        min_outputs=min_outputs, max_outputs=max_outputs
    )
    return NormalizedHBCAnsatz1D(
        displacement_left=displacement_left,
        network=network,
        network_input_normalizer=network_input_normalizer,
        hbc_ansatz_coordinate_normalizer=hbc_ansatz_coordinate_normalizer,
        hbc_ansatz_output_normalizer=hbc_ansatz_output_normalizer,
        hbc_ansatz_output_renormalizer=hbc_ansatz_output_renormalizer,
    )
