import torch
import torch.nn as nn

from parametricpinn.types import Tensor


class InputNormalizer(nn.Module):
    def __init__(self, min_inputs: Tensor, max_inputs: Tensor) -> None:
        super().__init__()
        self._min_inputs = min_inputs
        self._max_inputs = max_inputs
        self._input_ranges = max_inputs - min_inputs
        self._atol = torch.tensor([1e-7])

    def forward(self, x: Tensor) -> Tensor:
        return (
            (
                ((x - self._min_inputs) + self._atol)
                / (self._input_ranges + 2 * self._atol)
            )
            * 2.0
        ) - 1.0
