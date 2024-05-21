from typing import Optional

import gpytorch

from calibrationpinn.errors import (
    GPKernelNotImplementedError,
    GPMeanNotImplementedError,
)
from calibrationpinn.gps.base import GPMultivariateNormal, NamedParameters
from calibrationpinn.gps.kernels import Kernel, ScaledRBFKernel
from calibrationpinn.gps.means import ConstantMean, NonZeroMean, ZeroMean
from calibrationpinn.gps.normalizers import InputNormalizer
from calibrationpinn.gps.utility import validate_parameters_size
from calibrationpinn.types import Device, Tensor


class GP(gpytorch.models.ExactGP):
    def __init__(
        self,
        mean: ZeroMean | NonZeroMean,
        kernel: Kernel,
        input_normalizer: InputNormalizer,
        device: Device,
        train_x: Optional[Tensor] = None,
        train_y: Optional[Tensor] = None,
    ) -> None:
        train_x, train_y = self._preprocess_trainings_data(train_x, train_y, device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        super().__init__(train_x, train_y, likelihood)
        self.num_gps = 1
        self.num_hyperparameters = mean.num_hyperparameters + kernel.num_hyperparameters
        self._mean = mean
        self._kernel = kernel
        self._input_normalizer = input_normalizer
        self._is_zero_mean = isinstance(self._mean, ZeroMean)
        self._device = device

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        norm_x = self._input_normalizer(x)
        mean = self._mean(norm_x)
        covariance_matrix = self._kernel(norm_x, norm_x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance_matrix)

    def forward_mean(self, x: Tensor) -> Tensor:
        norm_x = self._input_normalizer(x)
        return self._mean(norm_x)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        norm_x_1 = self._input_normalizer(x_1)
        norm_x_2 = self._input_normalizer(x_2)
        lazy_covariance_matrix = self._kernel(norm_x_1, norm_x_2)
        return lazy_covariance_matrix.to_dense()

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
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

    def _preprocess_trainings_data(
        self, train_x: Optional[Tensor], train_y: Optional[Tensor], device: Device
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if train_x is not None and train_y is not None:
            train_x.to(device)
            train_y.to(device)
        return train_x, train_y


def create_gaussian_process(
    mean: str,
    kernel: str,
    min_inputs: Tensor,
    max_inputs: Tensor,
    device: Device,
    train_x: Optional[Tensor] = None,
    train_y: Optional[Tensor] = None,
) -> GP:
    mean_module = _create_mean(mean, device)
    kernel_module = _create_kernel(kernel, device)
    input_normalizer = _create_input_normalizer(min_inputs, max_inputs, device)
    return GP(
        mean=mean_module,
        kernel=kernel_module,
        input_normalizer=input_normalizer,
        device=device,
        train_x=train_x,
        train_y=train_y,
    )


def _create_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor, device: Device
) -> InputNormalizer:
    return InputNormalizer(min_inputs, max_inputs, device).to(device)


def _create_mean(mean: str, device: Device) -> ZeroMean | NonZeroMean:
    if mean == "zero":
        return ZeroMean(device).to(device)
    elif mean == "constant":
        return ConstantMean(device).to(device)
    else:
        raise GPMeanNotImplementedError(
            f"There is no implementation for the requested Gaussian process mean: {mean}."
        )


def _create_kernel(kernel: str, device: Device) -> Kernel:
    if kernel == "scaled_rbf":
        return ScaledRBFKernel(device).to(device)
    else:
        raise GPKernelNotImplementedError(
            f"There is no implementation for the requested Gaussian process kernel: {kernel}."
        )
