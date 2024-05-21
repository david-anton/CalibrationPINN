from dataclasses import dataclass

from calibrationpinn.bayesian.likelihood import Likelihood
from calibrationpinn.bayesian.prior import Prior
from calibrationpinn.calibration.config import CalibrationConfig


@dataclass
class MCMCConfig(CalibrationConfig):
    likelihood: Likelihood
    prior: Prior
    num_burn_in_iterations: int
