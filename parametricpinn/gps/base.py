from typing import Optional, Protocol, TypeAlias

import gpytorch

from parametricpinn.bayesian.prior import Prior
from parametricpinn.errors import UnvalidGPParametersError
from parametricpinn.types import Device, Tensor, TensorSize

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal
NamedParameters: TypeAlias = dict[str, Tensor]


class GaussianProcess(Protocol):
    num_gps: int
    num_hyperparameters: int
    _device: Device

    def __init__(
        self,
        device: Device,
        train_x: Optional[Tensor] = None,
        train_y: Optional[Tensor] = None,
    ) -> None:
        pass

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        pass

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        pass

    def set_parameters(self, parameters: Tensor) -> None:
        pass

    def get_named_parameters(self) -> NamedParameters:
        pass

    def get_uninformed_parameters_prior(self, device: Device) -> Prior:
        pass

    def to(self, device: Device) -> "GaussianProcess":
        pass


def validate_parameters_size(parameters: Tensor, valid_size: TensorSize) -> None:
    parameters_size = parameters.size()
    if parameters_size != valid_size:
        raise UnvalidGPParametersError(
            f"Parameter tensor has unvalid size {parameters_size}"
        )
