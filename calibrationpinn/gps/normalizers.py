import torch
import torch.nn as nn

from calibrationpinn.types import Device, Tensor


class InputNormalizer(nn.Module):
    def __init__(self, min_inputs: Tensor, max_inputs: Tensor, device: Device) -> None:
        super().__init__()
        self._min_inputs = min_inputs.to(device)
        self._max_inputs = max_inputs.to(device)
        self._input_ranges = max_inputs - min_inputs
        self._atol = 1e-7
        self._device = device

    def forward(self, x: Tensor) -> Tensor:
        denominator = self._input_ranges
        mask_division = torch.isclose(
            denominator,
            torch.zeros_like(denominator, device=self._device),
            atol=self._atol,
        )
        return torch.where(
            mask_division,
            torch.tensor([0.0], device=self._device),
            ((x - self._min_inputs) / denominator),
        )
