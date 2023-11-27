from typing import Optional, TypeAlias

import gpytorch
import torch

from parametricpinn.bayesian.prior import (
    Prior,
    create_mixed_independent_multivariate_distributed_prior,
)
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.statistics.distributions import (
    create_univariate_uniform_distribution,
)
from parametricpinn.types import Device, Tensor

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal


class ZeroMeanScaledRBFKernelGP(gpytorch.models.ExactGP):
    def __init__(
        self,
        device: Device,
        train_x: Optional[Tensor] = None,
        train_y: Optional[Tensor] = None,
    ) -> None:
        if train_x is not None and train_y is not None:
            train_x.to(device)
            train_y.to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        super(ZeroMeanScaledRBFKernelGP, self).__init__(train_x, train_y, likelihood)
        self.mean = gpytorch.means.ZeroMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_gps = 1

    def forward(self, x) -> GPMultivariateNormal:
        mean_x = self.mean(x)
        covariance_x = self.kernel(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covariance_x)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        lazy_tensor = self.kernel(x_1, x_2)
        return lazy_tensor.evaluate()

    def set_covariance_parameters(self, parameters: Tensor) -> None:
        valid_size = torch.Size([2])
        validate_parameters_size(parameters, valid_size)
        self.kernel.outputscale = parameters[0]
        self.kernel.base_kernel.lengthscale = parameters[1]

    def get_uninformed_covariance_parameters_prior(self, device: Device) -> Prior:
        outputscale_prior = create_univariate_uniform_distribution(
            lower_limit=0.0, upper_limit=10.0, device=device
        )
        lengthscale_prior = create_univariate_uniform_distribution(
            lower_limit=0.0, upper_limit=10.0, device=device
        )
        return create_mixed_independent_multivariate_distributed_prior(
            [outputscale_prior, lengthscale_prior]
        )
