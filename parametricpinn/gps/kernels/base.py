from typing import Protocol, TypeAlias
from dataclasses import dataclass

from gpytorch.lazy import LazyEvaluatedKernelTensor
from linear_operator.operators import LinearOperator

from parametricpinn.bayesian.prior import Prior
from parametricpinn.gps.base import NamedParameters
from parametricpinn.types import Device, Tensor

KernelOutput: TypeAlias = LazyEvaluatedKernelTensor | LinearOperator | Tensor


@dataclass
class KernelParameterPriorsConfig:
    pass


class Kernel(Protocol):
    num_hyperparameters: int

    def forward(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        pass

    def set_parameters(self, parameters: Tensor) -> None:
        pass

    def get_named_parameters(self) -> NamedParameters:
        pass

    def get_uninformed_parameters_prior(
        self, prior_config: KernelParameterPriorsConfig, device: Device
    ) -> Prior:
        pass

    def __call__(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        pass
