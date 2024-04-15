from typing import Callable, TypeAlias, Union, cast, overload

from parametricpinn.calibration.bayesianinference.mcmc import (
    EfficientNUTSConfig,
    HamiltonianConfig,
    MCMCConfig,
    MCMCOutput,
    MetropolisHastingsConfig,
    mcmc_efficientnuts,
    mcmc_hamiltonian,
    mcmc_metropolishastings,
    EMCEEConfig,
    mcmc_emcee
)
from parametricpinn.calibration.config import CalibrationConfig
from parametricpinn.calibration.leastsquares import (
    LeastSquaresConfig,
    LeastSquaresOutput,
    least_squares,
)
from parametricpinn.errors import CalibrationConfigError
from parametricpinn.types import Device

CalibrationOutput: TypeAlias = Union[MCMCOutput, LeastSquaresOutput]
CalibrationAlgorithmClosure: TypeAlias = Callable[[], CalibrationOutput]


@overload
def calibrate(calibration_config: MCMCConfig, device: Device) -> MCMCOutput:
    ...


@overload
def calibrate(
    calibration_config: LeastSquaresConfig, device: Device
) -> LeastSquaresOutput:
    ...


def calibrate(
    calibration_config: CalibrationConfig,
    device: Device,
) -> CalibrationOutput:
    calibration_algorithm = _create_calibration_algorithm(calibration_config, device)
    return calibration_algorithm()


def _create_calibration_algorithm(
    calibration_config: CalibrationConfig,
    device: Device,
) -> CalibrationAlgorithmClosure:
    if isinstance(calibration_config, MetropolisHastingsConfig):
        config_mh = cast(MetropolisHastingsConfig, calibration_config)
        print("MCMC algorithm used: Metropolis Hastings")

        def mcmc_mh_algorithm() -> MCMCOutput:
            return mcmc_metropolishastings(
                likelihood=config_mh.likelihood,
                prior=config_mh.prior,
                initial_parameters=config_mh.initial_parameters,
                cov_proposal_density=config_mh.cov_proposal_density,
                num_iterations=config_mh.num_iterations,
                num_burn_in_iterations=config_mh.num_burn_in_iterations,
                device=device,
            )

        return mcmc_mh_algorithm
    if isinstance(calibration_config, EMCEEConfig):
        config_emcee = cast(EMCEEConfig, calibration_config)
        print("MCMC algorithm used: emcee")

        def mcmc_emcee_algorithm() -> MCMCOutput:
            return mcmc_emcee(
                likelihood=config_emcee.likelihood,
                prior=config_emcee.prior,
                initial_parameters=config_emcee.initial_parameters,
                num_walkers=config_emcee.num_walkers,
                num_iterations=config_emcee.num_iterations,
                num_burn_in_iterations=config_emcee.num_burn_in_iterations,
                device=device,
            )

        return mcmc_emcee_algorithm
    elif isinstance(calibration_config, HamiltonianConfig):
        config_h = cast(HamiltonianConfig, calibration_config)
        print("MCMC algorithm used: Hamiltonian")

        def mcmc_hamiltonian_algorithm() -> MCMCOutput:
            return mcmc_hamiltonian(
                likelihood=config_h.likelihood,
                prior=config_h.prior,
                initial_parameters=config_h.initial_parameters.requires_grad_(True),
                num_leapfrog_steps=config_h.num_leabfrog_steps,
                leapfrog_step_sizes=config_h.leapfrog_step_sizes,
                num_iterations=config_h.num_iterations,
                num_burn_in_iterations=config_h.num_burn_in_iterations,
                device=device,
            )

        return mcmc_hamiltonian_algorithm
    elif isinstance(calibration_config, EfficientNUTSConfig):
        config_enuts = cast(EfficientNUTSConfig, calibration_config)
        print("MCMC algorithm used: Efficient NUTS")

        def mcmc_efficient_nuts_algorithm() -> MCMCOutput:
            return mcmc_efficientnuts(
                likelihood=config_enuts.likelihood,
                prior=config_enuts.prior,
                initial_parameters=config_enuts.initial_parameters.requires_grad_(True),
                leapfrog_step_sizes=config_enuts.leapfrog_step_sizes,
                num_iterations=config_enuts.num_iterations,
                num_burn_in_iterations=config_enuts.num_burn_in_iterations,
                max_tree_depth=config_enuts.max_tree_depth,
                device=device,
            )

        return mcmc_efficient_nuts_algorithm
    elif isinstance(calibration_config, LeastSquaresConfig):
        config_ls = cast(LeastSquaresConfig, calibration_config)
        print("Least squares algorithm used")

        def least_squares_algorithm() -> LeastSquaresOutput:
            return least_squares(
                ansatz=config_ls.ansatz,
                calibration_data=config_ls.calibration_data,
                initial_parameters=config_ls.initial_parameters,
                num_iterations=config_ls.num_iterations,
                residual_weights=config_ls.resdiual_weights,
                device=device,
            )

        return least_squares_algorithm
    else:
        raise CalibrationConfigError(
            f"There is no implementation for the requested calibration algorithm {calibration_config}."
        )
