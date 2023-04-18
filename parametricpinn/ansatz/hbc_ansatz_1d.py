# Standard library imports

# Third-party imports
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.types import Module, Tensor


class HBCAnsatz1D(nn.Module):
    def __init__(
        self, displacement_left: float, input_range_coordinate: float, network: Module
    ) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._input_range_coordinate = input_range_coordinate
        self._network = network

    def _boundary_data(self) -> float:
        return self._displacement_left

    def _distance_function(self, x_coordinate: Tensor) -> Tensor:
        return x_coordinate / self._input_range_coordinate

    def forward(self, x: Tensor) -> Tensor:
        x_coordinate = torch.unsqueeze(x[:, 0], 1)
        return self._boundary_data() + (
            self._distance_function(x_coordinate) * self._network(x)
        )
