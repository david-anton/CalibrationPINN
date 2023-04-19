import torch.nn as nn

from parametricpinn.types import Tensor


class OutputRenormalizer(nn.Module):
    def __init__(self, min_outputs: Tensor, max_outputs: Tensor) -> None:
        super().__init__()
        self._min_outputs = min_outputs
        self._max_outputs = max_outputs
        self._output_ranges = max_outputs - min_outputs

    def forward(self, x: Tensor) -> Tensor:
        return (((x + 1) / 2) * self._output_ranges) + self._min_outputs
