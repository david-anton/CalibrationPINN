from typing import TypeAlias, Union

import torch

from parametricpinn.errors import MixedDistributionError
from parametricpinn.types import Tensor, TorchUniformDist, TorchUniNormalDist

# There is no explicit univariate uniform distribution in Pytorch.
TorchUnivariateDistributions: TypeAlias = Union[TorchUniformDist, TorchUniNormalDist]


class MixedIndependetMultivariateDistribution:
    def __init__(self, distributions: list[TorchUnivariateDistributions]) -> None:
        self._distributions = distributions

    def _validate_sample(self, sample: Tensor) -> None:
        if sample.dim() != 1:
            raise MixedDistributionError(
                "Sample is expected to be a one-dimensional tensor."
            )
        if sample.size()[0] != len(self._distributions):
            raise MixedDistributionError(
                "The size of sample does not match the number of mixed distributions."
            )

    def log_prob(self, sample: Tensor) -> Tensor:
        self._validate_sample(sample)
        return torch.sum(
            torch.tensor(
                [
                    self._distributions[i].log_prob(sample[i])
                    for i in range(sample.size()[0])
                ]
            )
        )
