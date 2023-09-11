from dataclasses import dataclass

from parametricpinn.calibration.config import CalibrationConfig


@dataclass
class MCMCConfig(CalibrationConfig):
    num_burn_in_iterations: int
