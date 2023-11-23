from typing import TypeAlias

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
    def __init__(self, train_x=None, train_y=None) -> None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ZeroMeanScaledRBFKernelGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covariance_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x) -> GPMultivariateNormal:
        mean_x = self.mean_module(x)
        covariance_x = self.covariance_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covariance_x)

    def set_covariance_parameters(self, parameters: Tensor) -> None:
        valid_size = torch.Size([2])
        validate_parameters_size(parameters, valid_size)
        self.covariance_module.outputscale = parameters[0]
        self.covariance_module.lengthscale = parameters[1]

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
