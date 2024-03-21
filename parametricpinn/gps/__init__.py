from typing import TypeAlias

from parametricpinn.gps.gp import GP, create_gaussian_process
from parametricpinn.gps.kernels import RBFKernelParameterPriorConfig
from parametricpinn.gps.means import ConstantMeanParameterPriorConfig
from parametricpinn.gps.multioutputgp import IndependentMultiOutputGP
from parametricpinn.gps.prior import create_uninformed_gp_parameters_prior

GaussianProcess: TypeAlias = GP | IndependentMultiOutputGP

__all__ = [
    "GaussianProcess",
    "create_gaussian_process",
    "RBFKernelParameterPriorConfig",
    "ConstantMeanParameterPriorConfig",
    "IndependentMultiOutputGP",
    "create_uninformed_gp_parameters_prior",
]
