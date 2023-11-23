from typing import TypeAlias

import gpytorch
import torch

from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Tensor

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
