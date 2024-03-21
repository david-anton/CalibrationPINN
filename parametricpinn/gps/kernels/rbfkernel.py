import gpytorch
import torch

from parametricpinn.gps.base import NamedParameters
from parametricpinn.gps.kernels.base import Kernel, KernelOutput
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Device, Tensor


class RBFKernel(Kernel):
    def __init__(self, device: Device) -> None:
        super().__init__(num_hyperparameters=2)
        self._kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(
            device
        )
        self._device = device
        self._lower_limit_output_scale = 0.0
        self._lower_limit_length_scale = 0.0

    def forward(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        return self._kernel(x_1, x_2)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, torch.Size([self.num_hyperparameters]))
        output_scale = parameters[0]
        length_scale = parameters[1]
        if output_scale >= self._lower_limit_output_scale:
            self._kernel.outputscale = output_scale.clone().to(self._device)
        if length_scale >= self._lower_limit_length_scale:
            self._kernel.base_kernel.lengthscale = length_scale.clone().to(self._device)

    def get_named_parameters(self) -> NamedParameters:
        return {
            "output_scale": self._kernel.outputscale,
            "length_scale": self._kernel.base_kernel.lengthscale,
        }
