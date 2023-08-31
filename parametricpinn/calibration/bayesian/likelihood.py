from typing import Protocol

from parametricpinn.types import Tensor


class Likelihood(Protocol):
    def prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        pass
