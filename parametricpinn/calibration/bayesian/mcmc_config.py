from dataclasses import dataclass

from parametricpinn.types import Tensor


@dataclass
class MCMCConfig:
    parameter_names: tuple[str, ...]
    true_parameters: tuple[float, ...]
    prior_means: list[float]
    prior_stds: list[float]
    initial_parameters: Tensor
    num_iterations: int
