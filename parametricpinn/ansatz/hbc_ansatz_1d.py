import torch
import torch.nn as nn

from parametricpinn.types import Module, Tensor


class HBCAnsatz1D(nn.Module):
    def __init__(
        self, network: Module, displacement_left: float, range_coordinate: float
    ) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._range_coordinate = range_coordinate
        self._network = network

    def _boundary_data_func(self) -> float:
        return self._displacement_left

    def _distance_func(self, x_coor: Tensor) -> Tensor:
        return x_coor / self._range_coordinate

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return torch.unsqueeze(x[0], 0)
        return torch.unsqueeze(x[:, 0], 1)

    def forward(self, x: Tensor) -> Tensor:
        x_coor = self._extract_coordinates(x)
        return self._boundary_data_func() + (
            self._distance_func(x_coor) * self._network(x)
        )
