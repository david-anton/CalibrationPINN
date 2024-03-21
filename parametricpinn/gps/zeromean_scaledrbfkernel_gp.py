from typing import Optional

import gpytorch
import torch

from parametricpinn.bayesian.prior import (
    Prior,
    create_univariate_uniform_distributed_prior,
    multiply_priors,
)
from parametricpinn.gps.base import (
    GPMultivariateNormal,
    NamedParameters,
    validate_parameters_size,
)
from parametricpinn.gps.kernels import RBFKernel, Kernel
from parametricpinn.gps.means import ZeroMean
from parametricpinn.types import Device, Tensor
from parametricpinn.errors import GPKernelNotImplementedError


kernels = {"rbf": RBFKernel}


class ZeroMeanGP(gpytorch.models.ExactGP):
    def __init__(
        self,
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
        self._mean = ZeroMean(device)
        self._kernel = kernel
        self._device = device
        self.num_gps = 1
        self.num_hyperparameters = (
            self._mean.num_hyperparameters + self._kernel.num_hyperparameters
        )

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        mean = self._mean(x)
        covariance_matrix = self._kernel(x, x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance_matrix)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        lazy_covariance_matrix = self._kernel(x_1, x_2)
        return lazy_covariance_matrix.to_dense().to(self._device)

    def set_parameters(self, parameters: Tensor) -> None:
        self._kernel.set_parameters(parameters)

    def get_named_parameters(self) -> NamedParameters:
        return self._kernel.get_named_parameters()

    def get_uninformed_parameters_prior(
        self,
        upper_limit_output_scale: float,
        upper_limit_length_scale: float,
        device: Device,
        **kwargs: float,
    ) -> Prior:
        return self.get_uninformed_parameters_prior(
            upper_limit_output_scale=upper_limit_output_scale,
            upper_limit_length_scale=upper_limit_length_scale,
            device=device,
            **kwargs,
        )


def create_zero_mean_gaussian_process(
    kernel: str,
    device: Device,
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
) -> ZeroMeanGP:
    if kernel not in kernels.keys():
        raise GPKernelNotImplementedError(
            f"There is no implementation for the requested Gaussian process kernel {kernel}."
        )
    kernel_type = kernels[kernel]
    return ZeroMeanGP(
        kernel=kernel_type(device), device=device, train_x=train_x, train_y=train_y
    )
