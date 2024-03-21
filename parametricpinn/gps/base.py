from typing import TypeAlias

import gpytorch

from parametricpinn.types import Tensor

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal
NamedParameters: TypeAlias = dict[str, Tensor]
