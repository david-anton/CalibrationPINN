import gpytorch
import torch

from parametricpinn.gps.means.base import MeanOutput
from parametricpinn.types import Device, Tensor


class ZeroMean(torch.nn.Module):
    def __init__(self, device: Device) -> None:
        super().__init__()
        self._mean = gpytorch.means.ZeroMean().to(device)
        self._device = device
        self.num_hyperparameters = 0

    def forward(self, x: Tensor) -> MeanOutput:
        return self._mean(x)

    def __call__(self, x: Tensor) -> MeanOutput:
        return self.forward(x)
