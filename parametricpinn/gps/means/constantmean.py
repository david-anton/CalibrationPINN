import gpytorch
import torch

from parametricpinn.gps.base import NamedParameters
from parametricpinn.gps.means.base import MeanOutput, NonZeroMean
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Device, Tensor


class ConstantMean(NonZeroMean):
    def __init__(self, device: Device) -> None:
        super().__init__(num_hyperparameters=1)
        self._mean = gpytorch.means.ConstantMean().to(device)
        self._device = device

    def forward(self, x: Tensor) -> MeanOutput:
        return self._mean(x)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, torch.Size([self.num_hyperparameters]))
        constant_mean = parameters[0]
        self._mean.constant = constant_mean

    def get_named_parameters(self) -> NamedParameters:
        return {
            "constant_mean": self._mean.constant,
        }
