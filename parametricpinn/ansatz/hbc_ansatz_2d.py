import torch
import torch.nn as nn

from parametricpinn.settings import get_device
from parametricpinn.types import Module, Tensor

device = get_device()


class HBCAnsatz2D(nn.Module):
    def __init__(
        self,
        network: Module,
        displacement_x_right: float,
        displacement_y_bottom: float,
        range_coordinates: Tensor
    ) -> None:
        super().__init__()
        self._network = network
        self._boundary_data = torch.tensor(
            [displacement_x_right, displacement_y_bottom]
        ).to(device)
        self._range_coordinates = range_coordinates

    def _boundary_data_func(self) -> Tensor:
        return self._boundary_data

    def _distance_func(self, x_coor: Tensor) -> Tensor:
        return x_coor / self._range_coordinates

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x[0:2]
        return x[:, 0:2]

    def forward(self, x: Tensor) -> Tensor:
        x_coor = self._extract_coordinates(x)
        y = self._boundary_data_func() + (
            self._distance_func(x_coor) * self._network(x)
        )
        print(x)
        print(y)
        return y
