# Standard library imports

# Third-party imports
import torch.nn as nn

# Local library imports
from parametricpinn.types import Tensor


class InputNormalizer(nn.Module):
    def __init__(self, min_inputs: Tensor, max_inputs: Tensor) -> None:
        super().__init__()
        self._min_inputs = min_inputs
        self._max_inputs = max_inputs
        self._input_ranges = max_inputs - min_inputs

    def forward(self, x: Tensor) -> Tensor:
        return (((x - self._min_inputs) / self._input_ranges) * 2.0) - 1.0
