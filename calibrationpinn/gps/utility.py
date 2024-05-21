from typing import TypeAlias

import gpytorch
import torch

from calibrationpinn.errors import UnvalidGPParametersError
from calibrationpinn.types import Tensor, TensorSize

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal
NamedParameters: TypeAlias = dict[str, Tensor]


def validate_parameters_size(parameters: Tensor, valid_size: int | TensorSize) -> None:
    parameters_size = parameters.size()
    if isinstance(valid_size, int):
        valid_size = torch.Size([valid_size])
    if parameters_size != valid_size:
        raise UnvalidGPParametersError(
            f"Parameter tensor has unvalid size {parameters_size}"
        )
