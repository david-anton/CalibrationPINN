from typing import Callable, TypeAlias, cast

from parametricpinn.calibration.bayesian.likelihood import Likelihood
from parametricpinn.calibration.bayesian.mcmc import (
    EfficientNUTSConfig,
    HamiltonianConfig,
    MCMCConfig,
    MetropolisHastingsConfig,
    mcmc_efficientnuts,
    mcmc_hamiltonian,
    mcmc_metropolishastings,
)
from parametricpinn.calibration.bayesian.prior import Prior
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.errors import MCMCConfigError
from parametricpinn.types import Device, NPArray

MCMC_Algorithm_Output: TypeAlias = tuple[MomentsMultivariateNormal, NPArray]
MCMC_Algorithm_Closure: TypeAlias = Callable[[], MCMC_Algorithm_Output]


def calibrate(
    likelihood: Likelihood,
    prior: Prior,
    mcmc_config: MCMCConfig,
    device: Device,
) -> MCMC_Algorithm_Output:
    mcmc_algorithm = _create_mcmc_algorithm(
        likelihood=likelihood,
        prior=prior,
        mcmc_config=mcmc_config,
        device=device,
    )
    return mcmc_algorithm()


def _create_mcmc_algorithm(
    likelihood: Likelihood,
    prior: Prior,
    mcmc_config: MCMCConfig,
    device: Device,
) -> MCMC_Algorithm_Closure:
    if isinstance(mcmc_config, MetropolisHastingsConfig):
        config_mh = cast(MetropolisHastingsConfig, mcmc_config)
        print("MCMC algorithm used: Metropolis Hastings")

        def mcmc_mh_algorithm() -> MCMC_Algorithm_Output:
            return mcmc_metropolishastings(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_mh.initial_parameters,
                cov_proposal_density=config_mh.cov_proposal_density,
                num_iterations=config_mh.num_iterations,
                num_burn_in_iterations=config_mh.num_burn_in_iterations,
                device=device,
            )

        return mcmc_mh_algorithm

    elif isinstance(mcmc_config, HamiltonianConfig):
        config_h = cast(HamiltonianConfig, mcmc_config)
        print("MCMC algorithm used: Hamiltonian")

        def mcmc_hamiltonian_algorithm() -> MCMC_Algorithm_Output:
            return mcmc_hamiltonian(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_h.initial_parameters,
                num_leapfrog_steps=config_h.num_leabfrog_steps,
                leapfrog_step_sizes=config_h.leapfrog_step_sizes,
                num_iterations=config_h.num_iterations,
                num_burn_in_iterations=config_h.num_burn_in_iterations,
                device=device,
            )

        return mcmc_hamiltonian_algorithm
    elif isinstance(mcmc_config, EfficientNUTSConfig):
        config_enuts = cast(EfficientNUTSConfig, mcmc_config)
        print("MCMC algorithm used: Efficient NUTS")

        def mcmc_efficient_nuts_algorithm() -> MCMC_Algorithm_Output:
            return mcmc_efficientnuts(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_enuts.initial_parameters,
                leapfrog_step_sizes=config_enuts.leapfrog_step_sizes,
                num_iterations=config_enuts.num_iterations,
                num_burn_in_iterations=config_enuts.num_burn_in_iterations,
                max_tree_depth=config_enuts.max_tree_depth,
                device=device,
            )

        return mcmc_efficient_nuts_algorithm
    else:
        raise MCMCConfigError(
            f"There is no implementation for the requested MCMC algorithm {mcmc_config}."
        )
