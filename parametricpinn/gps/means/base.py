from abc import ABC, abstractmethod
from typing import TypeAlias

import torch

from parametricpinn.gps.base import NamedParameters
from parametricpinn.types import Device, Tensor

MeanOutput: TypeAlias = Tensor


class NonZeroMean(torch.nn.Module, ABC):
    def __init__(self, num_hyperparameters: int, device: Device) -> None:
        super().__init__()
        self.num_hyperparameters = num_hyperparameters
        self._device = device

    @abstractmethod
    def forward(self, x: Tensor) -> MeanOutput:
        pass

    @abstractmethod
    def set_parameters(self, parameters: Tensor) -> None:
        pass

    @abstractmethod
    def get_named_parameters(self) -> NamedParameters:
        pass

    def __call__(self, x: Tensor) -> MeanOutput:
        return self.forward(x)
