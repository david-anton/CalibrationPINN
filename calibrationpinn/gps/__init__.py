from typing import TypeAlias

from calibrationpinn.gps.gp import GP, create_gaussian_process
from calibrationpinn.gps.kernels import ScaledRBFKernelParameterPriorConfig
from calibrationpinn.gps.means import ConstantMeanParameterPriorConfig
from calibrationpinn.gps.multioutputgp import IndependentMultiOutputGP
from calibrationpinn.gps.prior import create_uninformed_gp_parameters_prior

GaussianProcess: TypeAlias = GP | IndependentMultiOutputGP

__all__ = [
    "GaussianProcess",
    "create_gaussian_process",
    "ScaledRBFKernelParameterPriorConfig",
    "ConstantMeanParameterPriorConfig",
    "IndependentMultiOutputGP",
    "create_uninformed_gp_parameters_prior",
]
