from typing import Callable, TypeAlias

import numpy as np
import scipy.stats
import torch
from scipy.stats._continuous_distns import norm_gen

from parametricpinn.types import Device, Module, NPArray, Tensor

NormalRV: TypeAlias = norm_gen
LikelihoodFunc: TypeAlias = Callable[[NPArray], float]


def mcmc_metropolishastings(likelihood: NormalRV, prior: NormalRV) -> NormalRV:
    unnormalized_posterior = None


def compile_likelihood_func(
    model: Module,
    coordinates: NPArray,
    data: NPArray,
    covariance_error: NPArray,
    device: Device,
) -> LikelihoodFunc:
    model.to(device)
    dim_data = data.shape[0]
    inverse_covariance_error = np.linalg.inv(covariance_error)

    def likelihood_func(parameters: NPArray) -> float:
        model_input = torch.concat(
            (torch.from_numpy(coordinates), torch.from_numpy(parameters)), dim=0
        ).to(device)
        prediction = model(model_input)
        return (
            1
            / (
                np.power(2 * np.pi, dim_data / 2)
                * np.power(np.linalg.det(covariance_error), 1 / 2)
            )
        ) * (
            np.exp(
                -1
                / 2
                * (prediction - data)
                * inverse_covariance_error
                * (prediction - data)
            ).detach().cpu()[0]
        )

    return likelihood_func


def plot_normal_random_variable(normal_rv: NormalRV):
    pass
