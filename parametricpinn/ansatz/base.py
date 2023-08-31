from typing import Protocol, TypeAlias, Union

import torch.nn as nn

from parametricpinn.network import BFFNN, FFNN
from parametricpinn.types import Tensor

Networks: TypeAlias = Union[FFNN, BFFNN]
StandardNetworks: TypeAlias = FFNN
BayesianNetworks: TypeAlias = BFFNN
FlattenedParameters: TypeAlias = Tensor


class AnsatzStrategy(Protocol):
    def __call__(self, x: Tensor, network: Networks) -> Tensor:
        pass


class StandardAnsatz(nn.Module):
    def __init__(
        self, network: StandardNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__()
        self.network = network
        self._ansatz_strategy = ansatz_strategy

    def forward(self, x: Tensor) -> Tensor:
        return self._ansatz_strategy(x, self.network)


class BayesianAnsatz(nn.Module):
    def __init__(
        self, network: BayesianNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__()
        self.network = network
        self._ansatz_strategy = ansatz_strategy

    def forward(self, x: Tensor) -> Tensor:
        return self._ansatz_strategy(x, self.network)
