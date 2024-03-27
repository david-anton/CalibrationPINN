from dataclasses import dataclass
from typing import TypeAlias

import torch

from parametricpinn.bayesian.prior import (
    Prior,
    create_multivariate_uniform_distributed_prior,
)
from parametricpinn.errors import GPKernelPriorNotImplementedError
from parametricpinn.types import Device


@dataclass
class ScaledRBFKernelParameterPriorConfig:
    limits_output_scale: tuple[float, float]
    limits_length_scale: tuple[float, float]


KernelParameterPriorConfig: TypeAlias = ScaledRBFKernelParameterPriorConfig


def create_uninformed_kernel_parameters_prior(
    config: KernelParameterPriorConfig, device: Device
) -> Prior:
    index_lower_limit = 0
    index_upper_limit = 1
    if isinstance(config, ScaledRBFKernelParameterPriorConfig):
        limits_output_scale = config.limits_output_scale
        limits_length_scale = config.limits_length_scale
        return create_multivariate_uniform_distributed_prior(
            lower_limits=torch.tensor(
                [
                    limits_output_scale[index_lower_limit],
                    limits_length_scale[index_lower_limit],
                    limits_length_scale[index_lower_limit],
                ],
                device=device,
            ),
            upper_limits=torch.tensor(
                [
                    limits_output_scale[index_upper_limit],
                    limits_length_scale[index_upper_limit],
                    limits_length_scale[index_upper_limit],
                ],
                device=device,
            ),
            device=device,
        )
    else:
        raise GPKernelPriorNotImplementedError(
            f"There is no implementation for the requested Gaussian process kernel parameters prior config: {config}."
        )
