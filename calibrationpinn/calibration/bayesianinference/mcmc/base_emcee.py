from typing import Callable, TypeAlias

import emcee
import numpy as np
import torch

from calibrationpinn.bayesian.likelihood import Likelihood
from calibrationpinn.bayesian.prior import Prior
from calibrationpinn.calibration.bayesianinference.mcmc.base import (
    Parameters,
    _log_unnormalized_posterior,
)
from calibrationpinn.errors import EMCEEConfigError
from calibrationpinn.types import Device, NPArray

LogProbFunc: TypeAlias = Callable[[NPArray], float]
Sampler: TypeAlias = emcee.EnsembleSampler
State: TypeAlias = emcee.State
Samples: TypeAlias = NPArray


def create_log_prob_func(
    likelihood: Likelihood, prior: Prior, device: Device
) -> LogProbFunc:
    log_unnorm_posterior = _log_unnormalized_posterior(likelihood, prior)

    def log_prob_func(parameters: NPArray) -> float:
        parameters_torch = torch.from_numpy(parameters).to(device)
        log_prob = log_unnorm_posterior(parameters_torch)
        return log_prob.detach().cpu().item()

    return log_prob_func


def validate_initial_parameters(
    initial_parameters: Parameters, num_walkers: int
) -> None:
    num_initial_walkers = initial_parameters.size()[0]
    if not num_initial_walkers == num_walkers:
        raise EMCEEConfigError(
            f"First dimension of initial parameters is {num_initial_walkers} \
            and does not match with number of walkers which is {num_walkers}."
        )


def validate_stretch_scale(stretch_scale: float) -> None:
    if stretch_scale <= 1.0:
        raise EMCEEConfigError(f"The stretch scale must be larger than 1.0.")


def print_mean_acceptance_ratio(sampler: Sampler) -> None:
    acceptance_ratio = sampler.acceptance_fraction
    mean_acceptance_ratio = np.mean(acceptance_ratio)
    print(f"Mean acceptance ratio: {round(mean_acceptance_ratio, 4)}")
