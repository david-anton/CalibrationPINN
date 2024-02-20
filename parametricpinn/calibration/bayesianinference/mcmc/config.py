from dataclasses import dataclass

from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.config import CalibrationConfig


@dataclass
class MCMCConfig(CalibrationConfig):
    likelihood: Likelihood
    prior: Prior
    num_burn_in_iterations: int
