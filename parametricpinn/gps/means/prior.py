from dataclasses import dataclass

from parametricpinn.bayesian.prior import (
    Prior,
    create_univariate_uniform_distributed_prior,
)
from parametricpinn.errors import GPMeanPriorNotImplementedError
from parametricpinn.types import Device


@dataclass
class MeanParameterPriorConfig:
    pass


class ConstantMeanParameterPriorConfig(MeanParameterPriorConfig):
    limits_constant_mean: tuple[float, float]


def create_uninformed_mean_parameters_prior(
    config: MeanParameterPriorConfig, device: Device
) -> Prior:
    index_lower_limit = 0
    index_upper_limit = 1
    if isinstance(config, ConstantMeanParameterPriorConfig):
        limits_constant_mean = config.limits_constant_mean
        return create_univariate_uniform_distributed_prior(
            lower_limit=limits_constant_mean[index_lower_limit],
            upper_limit=limits_constant_mean[index_upper_limit],
            device=device,
        )
    else:
        raise GPMeanPriorNotImplementedError(
            f"There is no implementation for the requested Gaussian process mean parameters prior config: {config}."
        )
