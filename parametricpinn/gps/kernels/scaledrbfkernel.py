import gpytorch
import torch

from parametricpinn.gps.base import NamedParameters
from parametricpinn.gps.kernels.base import Kernel, KernelOutput
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Device, Tensor


class ScaledRBFKernel(Kernel):
    def __init__(self, device: Device) -> None:
        super().__init__(
            num_hyperparameters=2,
            device=device,
        )
        self._kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        ).to(device)
        self._lower_limit_output_scale = 0.0
        self._lower_limit_length_scale = 0.0

    def forward(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        return self._kernel(x_1, x_2)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        output_scale = parameters[0]
        length_scales = parameters[1:]
        self._set_output_scale(output_scale)
        self._set_length_scales(length_scales)

    def _set_output_scale(self, output_scale: Tensor) -> None:
        if output_scale >= self._lower_limit_output_scale:
            self._kernel.outputscale = output_scale.clone().to(self._device)

    def _set_length_scales(self, length_scales: Tensor) -> None:
        current_length_scales = self._kernel.base_kernel.lengthscale
        updated_length_scales = torch.where(
            length_scales >= self._lower_limit_length_scale,
            length_scales,
            current_length_scales,
        )
        self._kernel.base_kernel.lengthscale = updated_length_scales.clone().to(
            self._device
        )

    def get_named_parameters(self) -> NamedParameters:
        return {
            "output_scale": self._kernel.outputscale,
            "length_scale": self._kernel.base_kernel.lengthscale,
        }
