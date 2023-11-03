from .empiricalconvergencetest import calculate_empirical_convegrence_order
from .errornorms import h01_error, infinity_error, l2_error, relative_l2_error
from .plot import plot_error_convergence_analysis

__all__ = [
    "h01_error",
    "calculate_empirical_convegrence_order",
    "infinity_error",
    "l2_error",
    "relative_l2_error",
    "plot_error_convergence_analysis",
]
