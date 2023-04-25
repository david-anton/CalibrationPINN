import torch
import torch.nn as nn

from parametricpinn.types import Module, Tensor


class HBCAnsatz2D(nn.Module):
    def __init__(
        self,
        network: Module,
        displacement_x_right: float,
        displacement_y_bottom: float,
        range_coordinate_x: float,
        range_coordinate_y: float,
    ) -> None:
        super().__init__()
        self._network = network
        self._displacement_x_right = displacement_x_right
        self._displacement_y_bottom = displacement_y_bottom
        self._range_coordinates = torch.tensor([range_coordinate_x, range_coordinate_y])

    def _boundary_data(self) -> Tensor:
        return torch.tensor([self._displacement_x_right, self._displacement_y_bottom])

    def _distance_function(self, x_coor: Tensor) -> Tensor:
        return x_coor / self._range_coordinates

    def _extract_coordinates(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x[0:2]
        return x[:, 0:2]

    def forward(self, x: Tensor) -> Tensor:
        x_coor = self._extract_coordinates(x)
        return self._boundary_data() + (
            self._distance_function(x_coor) * self._network(x_coor)
        )
