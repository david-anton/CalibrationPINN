from typing import Optional

import gpytorch
import torch

from parametricpinn.errors import GPKernelNotImplementedError, GPMeanNotImplementedError
from parametricpinn.gps.base import GPMultivariateNormal, NamedParameters
from parametricpinn.gps.kernels import Kernel, RBFKernel
from parametricpinn.gps.means import ConstantMean, NonZeroMean, ZeroMean
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Device, Tensor


class GaussianProcess(gpytorch.models.ExactGP):
    def __init__(
        self,
        mean: ZeroMean | NonZeroMean,
        kernel: Kernel,
        device: Device,
        train_x: Optional[Tensor] = None,
        train_y: Optional[Tensor] = None,
    ) -> None:
        if train_x is not None and train_y is not None:
            train_x.to(device)
            train_y.to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        super().__init__(train_x, train_y, likelihood)
        self._mean = mean
        self._kernel = kernel
        self._device = device
        self._is_zero_mean = isinstance(self._mean, ZeroMean)
        self.num_gps = 1
        self.num_hyperparameters = (
            self._mean.num_hyperparameters + self._kernel.num_hyperparameters
        )

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        mean = self._mean(x)
        covariance_matrix = self._kernel(x, x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance_matrix)

    def forward_mean(self, x: Tensor) -> Tensor:
        return self._mean(x)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        lazy_covariance_matrix = self._kernel(x_1, x_2)
        return lazy_covariance_matrix.to_dense().to(self._device)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, torch.Size([self.num_hyperparameters]))
        if self._is_zero_mean:
            self._kernel.set_parameters(parameters)
        else:
            num_parameters_mean = self._mean.num_hyperparameters
            self._mean.set_parameters(parameters[:num_parameters_mean])
            self._kernel.set_parameters(parameters[num_parameters_mean:])

    def get_named_parameters(self) -> NamedParameters:
        parameters_kernel = self._kernel.get_named_parameters()
        if self._is_zero_mean:
            return parameters_kernel
        else:
            parameters_mean = self._mean.get_named_parameters()
            return parameters_mean | parameters_kernel


def create_gaussian_process(
    mean: str,
    kernel: str,
    device: Device,
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
) -> GaussianProcess:
    mean_module = _create_mean_module(mean, device)
    kernel_module = _create_kernel_module(kernel, device)
    return GaussianProcess(
        mean=mean_module,
        kernel=kernel_module,
        device=device,
        train_x=train_x,
        train_y=train_y,
    )


def _create_mean_module(mean: str, device: Device) -> ZeroMean | NonZeroMean:
    if mean == "zero":
        return ZeroMean(device)
    elif mean == "constant":
        return ConstantMean(device)
    else:
        raise GPMeanNotImplementedError(
            f"There is no implementation for the requested Gaussian process mean: {mean}."
        )


def _create_kernel_module(kernel: str, device: Device) -> Kernel:
    if kernel == "rbf":
        return RBFKernel(device)
    else:
        raise GPKernelNotImplementedError(
            f"There is no implementation for the requested Gaussian process kernel: {kernel}."
        )
