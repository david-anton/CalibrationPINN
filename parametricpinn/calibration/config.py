from dataclasses import dataclass

from parametricpinn.types import Tensor


@dataclass
class CalibrationConfig:
    initial_parameters: Tensor
    num_iterations: int
