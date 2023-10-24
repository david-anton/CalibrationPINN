import torch.nn as nn

from parametricpinn.types import Tensor


class HBCAnsatzNormalizer(nn.Module):
    def __init__(self, min_values: Tensor, max_values: Tensor) -> None:
        super().__init__()
        self._value_ranges = max_values - min_values

    def forward(self, input: Tensor) -> Tensor:
        return input / self._value_ranges


class HBCAnsatzRenormalizer(nn.Module):
    def __init__(self, min_values: Tensor, max_values: Tensor) -> None:
        super().__init__()
        self._value_ranges = max_values - min_values

    def forward(self, normalized_input: Tensor) -> Tensor:
        return normalized_input * self._value_ranges
