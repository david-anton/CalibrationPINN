from typing import Optional, Protocol, TypeAlias

import gpytorch

from parametricpinn.bayesian.prior import Prior
from parametricpinn.types import Device, Tensor

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal


class GaussianProcess(Protocol):
    def __init__(
        self,
        device: Device,
        train_x: Optional[Tensor] = None,
        train_y: Optional[Tensor] = None,
    ) -> None:
        self.num_gps: int
        self.num_hyperparameters: int
        self._device: Device

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        pass

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        pass

    def set_parameters(self, parameters: Tensor) -> None:
        pass

    def get_uninformed_parameters_prior(self, device: Device) -> Prior:
        pass

    def to(self, device: Device) -> "GaussianProcess":
        pass
