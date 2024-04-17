from dataclasses import dataclass
from typing import Callable, TypeAlias

import emcee

from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.bayesianinference.mcmc.base import (
    MCMCOutput,
    Parameters,
)
from parametricpinn.calibration.bayesianinference.mcmc.base_emcee import (
    Samples,
    State,
    create_log_prob_func,
    validate_initial_parameters,
    validate_stretch_scale,
    print_mean_acceptance_ratio,
)
from parametricpinn.calibration.bayesianinference.mcmc.config import MCMCConfig
from parametricpinn.statistics.utility import (
    determine_moments_of_multivariate_normal_distribution,
)
from parametricpinn.types import Device, NPArray

MCMCEMCEEFunc: TypeAlias = Callable[
    [
        Likelihood,
        Prior,
        Parameters,
        float,
        int,
        int,
        int,
        Device,
    ],
    MCMCOutput,
]


@dataclass
class EMCEEConfig(MCMCConfig):
    stretch_scale: float
    num_walkers: int
    algorithm_name = "emcee"


def mcmc_emcee(
    likelihood: Likelihood,
    prior: Prior,
    initial_parameters: Parameters,
    stretch_scale: float,
    num_walkers: int,
    num_iterations: int,
    num_burn_in_iterations: int,
    device: Device,
) -> MCMCOutput:
    validate_initial_parameters(initial_parameters, num_walkers)
    validate_stretch_scale(stretch_scale)
    num_parameters = initial_parameters.size()[1]
    log_prob_func = create_log_prob_func(likelihood, prior, device)

    sampler = emcee.EnsembleSampler(
        nwalkers=num_walkers,
        ndim=num_parameters,
        log_prob_fn=log_prob_func,
        moves=emcee.moves.StretchMove(a=stretch_scale),
    )

    def _run_burn_in_phase() -> State:
        state = sampler.run_mcmc(
            initial_state=initial_parameters.detach().clone().cpu().numpy(),
            nsteps=num_burn_in_iterations,
        )
        sampler.reset()
        return state

    def _run_production_phase(burn_in_state: State) -> Samples:
        _ = sampler.run_mcmc(initial_state=burn_in_state, nsteps=num_iterations)
        return sampler.get_chain(flat=True)

    def run_mcmc() -> Samples:
        burn_in_state = _run_burn_in_phase()
        return _run_production_phase(burn_in_state)

    samples = run_mcmc()
    moments = determine_moments_of_multivariate_normal_distribution(samples)
    print_mean_acceptance_ratio(sampler)

    return moments, samples
