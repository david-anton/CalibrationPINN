# Standard library imports

# Third-party imports
import torch.nn as nn

# Local library imports
from parametricpinn.ansatz.normalization import InputNormalizer, OutputRenormalizer
from parametricpinn.types import Tensor, Module


class NormalizedHBCAnsatz1D(nn.Module):
    def __init__(
        self,
        network: Module,
        input_normalizer: Module,
        output_renormalizer: Module,
        input_range_coordinate: Tensor,
    ):
        super().__init__()
        self._network = network
        self._input_normalizer = input_normalizer
        self._output_renormalizer = output_renormalizer
        self._input_range_coordinate = input_range_coordinate

    def _boundary_data(self, coordinate: Tensor) -> float:
        return -1.0

    def _distance_function(self, coordinate: Tensor) -> Tensor:
        return coordinate / self._input_range_coordinate

    def forward(self, x: Tensor) -> Tensor:
        num_inputs = x.shape[0]
        x_coor = x[:, 0].view((num_inputs, 1))
        norm_x = self._input_normalizer(x)

        norm_y = self._boundary_data(x_coor) + (
            self._distance_function(x_coor) * self._network(norm_x)
        )
        renorm_y = self._output_renormalizer(norm_y)
        return renorm_y


def create_normalized_HBC_ansatz_1D(
    network: Module,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
) -> NormalizedHBCAnsatz1D:
    input_normalizer = InputNormalizer(min_inputs, max_inputs)
    output_renormalizer = OutputRenormalizer(min_outputs, max_outputs)
    input_range_coordinate = max_inputs[0] - min_inputs[0]
    return NormalizedHBCAnsatz1D(
        network=network,
        input_normalizer=input_normalizer,
        output_renormalizer=output_renormalizer,
        input_range_coordinate=input_range_coordinate,
    )
