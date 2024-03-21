from typing import TypeAlias

import gpytorch
import torch

from parametricpinn.bayesian.prior import (
    Prior,
    create_univariate_uniform_distributed_prior,
    multiply_priors,
)
from parametricpinn.gps.base import NamedParameters, validate_parameters_size
from parametricpinn.gps.kernels.base import KernelOutput, KernelParameterPriorsConfig
from parametricpinn.types import Device, Tensor


class RBFKernelParameterPriorConfig(KernelParameterPriorsConfig):
    upper_limit_output_scale: float
    upper_limit_length_scale: float


class RBFKernel(torch.nn.Module):
    def __init__(self, device: Device) -> None:
        super().__init__()
        self._kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()).to(
            device
        )
        self._device = device
        self._lower_limit_output_scale = 0.0
        self._lower_limit_length_scale = 0.0
        self.num_hyperparameters = 2

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

    def get_uninformed_parameters_prior(
        self,
        priot_config: RBFKernelParameterPriorConfig,
        device: Device,
    ) -> Prior:
        output_scale_prior = create_univariate_uniform_distributed_prior(
            lower_limit=self._lower_limit_output_scale,
            upper_limit=priot_config.upper_limit_output_scale,
            device=device,
        )
        length_scale_prior = create_univariate_uniform_distributed_prior(
            lower_limit=self._lower_limit_length_scale,
            upper_limit=priot_config.upper_limit_length_scale,
            device=device,
        )
        return multiply_priors([output_scale_prior, length_scale_prior])

    def __call__(self, x_1: Tensor, x_2: Tensor) -> KernelOutput:
        return self.forward(x_1, x_2)
