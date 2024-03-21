from typing import TypeAlias

from parametricpinn.bayesian.prior import Prior, multiply_priors
from parametricpinn.gps.kernels import create_uninformed_kernel_parameters_prior
from parametricpinn.gps.kernels.prior import KernelParameterPriorConfig
from parametricpinn.gps.means import create_uninformed_mean_parameters_prior
from parametricpinn.gps.means.prior import MeanParameterPriorConfig
from parametricpinn.types import Device

GPParametersPriorConfig: TypeAlias = tuple[
    MeanParameterPriorConfig, KernelParameterPriorConfig
]
GPParametersPriorConfigList: TypeAlias = list[GPParametersPriorConfig]


def create_uninformed_gp_parameters_prior(
    prior_config: GPParametersPriorConfig | GPParametersPriorConfigList, device: Device
) -> Prior:
    if not isinstance(prior_config, list):
        return _create_uninformed_parameters_prior_for_one_gp(prior_config, device)
    else:
        priors = [
            _create_uninformed_parameters_prior_for_one_gp(config, device)
            for config in prior_config
        ]
        return multiply_priors(priors)


def _create_uninformed_parameters_prior_for_one_gp(
    prior_config: GPParametersPriorConfig, device: Device
) -> Prior:
    mean_prior_config = prior_config[0]
    kernel_prior_config = prior_config[1]
    mean_prior = create_uninformed_mean_parameters_prior(mean_prior_config, device)
    kernel_prior = create_uninformed_kernel_parameters_prior(
        kernel_prior_config, device
    )
    priors = [mean_prior, kernel_prior]
    return multiply_priors(priors)
