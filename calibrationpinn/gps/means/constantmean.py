import gpytorch

from calibrationpinn.gps.base import NamedParameters
from calibrationpinn.gps.means.base import MeanOutput, NonZeroMean
from calibrationpinn.gps.utility import validate_parameters_size
from calibrationpinn.types import Device, Tensor


class ConstantMean(NonZeroMean):
    def __init__(self, device: Device) -> None:
        super().__init__(
            num_hyperparameters=1,
            device=device,
        )
        self._mean = gpytorch.means.ConstantMean().to(device)

    def forward(self, x: Tensor) -> MeanOutput:
        return self._mean(x)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        constant_mean = parameters[0]
        self._mean.constant = constant_mean

    def get_named_parameters(self) -> NamedParameters:
        return {
            "constant_mean": self._mean.constant,
        }
