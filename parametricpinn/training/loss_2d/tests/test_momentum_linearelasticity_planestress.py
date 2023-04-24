import torch
import torch.nn as nn

from parametricpinn.types import Tensor


class FakeAnsatzPlaneStress(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._constants = torch.tensor([8 / 39, -44 / 39])

    def forward(self, x: Tensor) -> Tensor:
        coor_x = x[0]
        coor_y = x[1]
        return self._constants * torch.tensor(
            [coor_x**2 * coor_y, coor_x * coor_y**2]
        )
