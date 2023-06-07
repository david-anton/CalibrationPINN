import torch.nn as nn

from parametricpinn.types import Tensor


class HBCAnsatzNormalizer(nn.Module):
    def __init__(self, min_values: Tensor, max_values: Tensor) -> None:
        super().__init__()
        self._value_ranges = max_values - min_values
        print("################################")
        print("################################")
        print(f"min_values {min_values.get_device()}")
        print(f"max_values {max_values.get_device()}")
        print(f"value_ranges {self._value_ranges.get_device()}")
        print("################################")
        print("################################")

    def forward(self, x: Tensor) -> Tensor:
        return x / self._value_ranges


class HBCAnsatzRenormalizer(nn.Module):
    def __init__(self, min_values: Tensor, max_values: Tensor) -> None:
        super().__init__()
        self._value_ranges = max_values - min_values
        print("################################")
        print("################################")
        print(f"min_values {min_values.get_device()}")
        print(f"max_values {max_values.get_device()}")
        print(f"value_ranges {self._value_ranges.get_device()}")
        print("################################")
        print("################################")

    def forward(self, normalized_x: Tensor) -> Tensor:
        return normalized_x * self._value_ranges
