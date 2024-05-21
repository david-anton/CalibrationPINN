from dataclasses import dataclass
from typing import TypeAlias

from calibrationpinn.bayesian.prior import (
    Prior,
    create_univariate_uniform_distributed_prior,
)
from calibrationpinn.errors import GPMeanPriorNotImplementedError
from calibrationpinn.types import Device


@dataclass
class ConstantMeanParameterPriorConfig:
    limits_constant_mean: tuple[float, float]


MeanParameterPriorConfig: TypeAlias = ConstantMeanParameterPriorConfig


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
