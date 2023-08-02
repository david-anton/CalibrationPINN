from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.types import Tensor, TorchMultiNormalDist

UnnormalizedPosterior: TypeAlias = Callable[[Tensor], Tensor]


def compile_unnnormalized_posterior(
    likelihood: LikelihoodFunc, prior: TorchMultiNormalDist
) -> UnnormalizedPosterior:
    def _unnormalized_posterior(parameters: Tensor) -> Tensor:
        return likelihood(parameters) * torch.pow(10, prior.log_prob(parameters))

    return _unnormalized_posterior
