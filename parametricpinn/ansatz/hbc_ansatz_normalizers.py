import torch.nn as nn

from parametricpinn.types import Tensor


class HBCAnsatzCoordinatesNormalizer(nn.Module):
    def __init__(self, min_coordinates: Tensor, max_coordinates: Tensor) -> None:
        super().__init__()
        self._coordinate_ranges = max_coordinates - min_coordinates

    def forward(self, x: Tensor) -> Tensor:
        return x / self._coordinate_ranges


class HBCAnsatzOutputNormalizer(nn.Module):
    def __init__(self, min_outputs: Tensor, max_outputs: Tensor) -> None:
        super().__init__()
        self._output_ranges = max_outputs - min_outputs

    def forward(self, x: Tensor) -> Tensor:
        return x / self._output_ranges


class HBCAnsatzOutputRenormalizer(nn.Module):
    def __init__(self, min_outputs: Tensor, max_outputs: Tensor) -> None:
        super().__init__()
        self._output_ranges = max_outputs - min_outputs

    def forward(self, x: Tensor) -> Tensor:
        return x * self._output_ranges
