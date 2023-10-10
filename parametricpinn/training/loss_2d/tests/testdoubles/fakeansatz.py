import torch
from torch import nn

from parametricpinn.types import Module, Tensor


class FakeAnsatz(nn.Module):
    def __init__(
        self, constant_displacement_x: float, constant_displacement_y: float
    ) -> None:
        super().__init__()
        self._constants = torch.tensor(
            [constant_displacement_x, constant_displacement_y]
        )

    def forward(self, x_coordinates: Tensor) -> Tensor:
        return self._constants * calculate_output(x_coordinates)


class TransformedFakeAnsatz(nn.Module):
    def __init__(
        self, constant_displacement_x: float, constant_displacement_y: float
    ) -> None:
        super().__init__()
        self._constants = torch.tensor(
            [constant_displacement_x, constant_displacement_y]
        )

    def forward(self, x_coordinates: Tensor, x_parameetrs: Tensor) -> Tensor:
        return self._constants * calculate_output(x_coordinates)


def calculate_output(x_coordinates: Tensor) -> Tensor:
    coor_x = torch.unsqueeze(x_coordinates[0], dim=0)
    coor_y = torch.unsqueeze(x_coordinates[1], dim=0)
    return torch.concat((coor_x**2 * coor_y, coor_x * coor_y**2), dim=0)
