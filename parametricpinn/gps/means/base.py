from typing import Protocol, TypeAlias

from parametricpinn.bayesian.prior import Prior
from parametricpinn.gps.base import NamedParameters
from parametricpinn.types import Device, Tensor

MeanOutput: TypeAlias = Tensor


class Mean(Protocol):
    def forward(self, x: Tensor) -> MeanOutput:
        pass

    def set_parameters(self, parameters: Tensor) -> None:
        pass

    def get_named_parameters(self) -> NamedParameters:
        pass

    def get_uninformed_parameters_prior(self, device: Device, **kwargs: float) -> Prior:
        pass

    def __call__(self, x_1: Tensor, x_2: Tensor) -> MeanOutput:
        pass
