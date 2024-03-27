from typing import TypeAlias

from parametricpinn.bayesian.prior import Prior, multiply_priors
from parametricpinn.gps.kernels import create_uninformed_kernel_parameters_prior
from parametricpinn.gps.kernels.prior import KernelParameterPriorConfig
from parametricpinn.gps.means import create_uninformed_mean_parameters_prior
from parametricpinn.gps.means.prior import MeanParameterPriorConfig
from parametricpinn.types import Device
from parametricpinn.errors import UnvalidGPPriorConfigError

ZeroMeanGPParametersPriorConfig: TypeAlias = tuple[KernelParameterPriorConfig]
GPParametersPriorConfig: TypeAlias = tuple[
    MeanParameterPriorConfig, KernelParameterPriorConfig
]
GPParametersPriorConfigTuple: TypeAlias = tuple[
    ZeroMeanGPParametersPriorConfig | GPParametersPriorConfig, ...
]


def create_uninformed_gp_parameters_prior(
    prior_config: GPParametersPriorConfigTuple, device: Device
) -> Prior:
    priors = [
        _create_uninformed_parameters_prior_for_one_gp(config, device)
        for config in prior_config
    ]
    if len(priors) == 1:
        return priors[0]
    else:
        return multiply_priors(priors)


def _create_uninformed_parameters_prior_for_one_gp(
    prior_config: ZeroMeanGPParametersPriorConfig | GPParametersPriorConfig,
    device: Device,
) -> Prior:
    if len(prior_config) == 1 and isinstance(
        prior_config[0], KernelParameterPriorConfig
    ):
        return create_uninformed_kernel_parameters_prior(prior_config[0], device)
    elif (
        len(prior_config) == 2
        and isinstance(prior_config[0], MeanParameterPriorConfig)
        and isinstance(prior_config[1], KernelParameterPriorConfig)
    ):
        mean_prior_config = prior_config[0]
        kernel_prior_config = prior_config[1]
        mean_prior = create_uninformed_mean_parameters_prior(mean_prior_config, device)
        kernel_prior = create_uninformed_kernel_parameters_prior(
            kernel_prior_config, device
        )
        priors = [mean_prior, kernel_prior]
        return multiply_priors(priors)
    else:
        raise UnvalidGPPriorConfigError(
            "There is no implementation for the requested prior of GP hyperparameters."
        )
