from typing import TypeAlias

import gpytorch

from parametricpinn.errors import UnvalidGPParametersError
from parametricpinn.types import Tensor, TensorSize

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal
NamedParameters: TypeAlias = dict[str, Tensor]


def validate_parameters_size(parameters: Tensor, valid_size: TensorSize) -> None:
    parameters_size = parameters.size()
    if parameters_size != valid_size:
        raise UnvalidGPParametersError(
            f"Parameter tensor has unvalid size {parameters_size}"
        )
