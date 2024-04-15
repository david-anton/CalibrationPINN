from dataclasses import dataclass

import emcee
import torch

from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.bayesianinference.mcmc.base import (
    MCMCOutput,
    Parameters,
    _log_unnormalized_posterior,
)
from parametricpinn.calibration.bayesianinference.mcmc.config import MCMCConfig
from parametricpinn.statistics.utility import (
    determine_moments_of_multivariate_normal_distribution,
)
from parametricpinn.types import Device, NPArray


@dataclass
class EMCEEConfig(MCMCConfig):
    num_walkers: int
    algorithm_name = "emcee"


def mcmc_emcee(
    likelihood: Likelihood,
    prior: Prior,
    initial_parameters: Parameters,
    num_walkers: int,
    num_iterations: int,
    num_burn_in_iterations: int,
    device: Device,
) -> MCMCOutput:
    log_unnorm_posterior = _log_unnormalized_posterior(likelihood, prior)
    num_parameters = len(initial_parameters)
    populated_initial_parameters = initial_parameters.repeat((num_walkers, 1))

    def log_prob_func(parameters: NPArray) -> float:
        parameters_torch = torch.from_numpy(parameters).to(device)
        log_prob = log_unnorm_posterior(parameters_torch)
        return log_prob.detach().cpu().item()

    sampler = emcee.EnsembleSampler(
        nwalkers=num_walkers, ndim=num_parameters, log_prob_fn=log_prob_func
    )

    # Burn-in phase
    state_burnin = sampler.run_mcmc(
        initial_state=populated_initial_parameters, nsteps=num_burn_in_iterations
    )
    sampler.reset()

    # Production phase
    _ = sampler.run_mcmc(initial_state=state_burnin, nsteps=num_iterations)
    samples = sampler.get_chain(flat=True)

    # Postprocessing
    moments = determine_moments_of_multivariate_normal_distribution(samples)
    acceptance_ratio = sampler.acceptance_fraction
    print(f"Acceptance ratio: {round(acceptance_ratio, 4)}")

    return moments, samples
