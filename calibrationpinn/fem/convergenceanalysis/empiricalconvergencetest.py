import numpy as np


def calculate_empirical_convegrence_order(
    results: list[float], reduction_factor: float
) -> float:
    result_h = results[0]
    result_h2 = results[1]
    result_h4 = results[2]
    convergence_order = (
        1
        / np.log(1 / reduction_factor)
        * np.log((result_h - result_h2) / (result_h2 - result_h4))
    )
    return convergence_order
