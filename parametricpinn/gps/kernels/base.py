from abc import ABC, abstractmethod
from typing import TypeAlias

import torch
from gpytorch.lazy import LazyEvaluatedKernelTensor
from linear_operator.operators import LinearOperator

from parametricpinn.gps.base import NamedParameters
from parametricpinn.types import Device, Tensor

KernelOutput: TypeAlias = LazyEvaluatedKernelTensor | LinearOperator | Tensor


class Kernel(torch.nn.Module, ABC):
    def __init__(self, num_hyperparameters: int, device: Device) -> None:
        super().__init__()
        self.num_hyperparameters = num_hyperparameters
        self._device = device

    @abstractmethod
    def forward(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        pass

    @abstractmethod
    def set_parameters(self, parameters: Tensor) -> None:
        pass

    @abstractmethod
    def get_named_parameters(self) -> NamedParameters:
        pass

    def __call__(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        return self.forward(x_1, x_2)
