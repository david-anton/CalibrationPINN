from dataclasses import dataclass

from parametricpinn.types import Tensor


@dataclass
class MCMCConfig:
    parameter_names: tuple[str, ...]
    true_parameters: tuple[float, ...]
    initial_parameters: Tensor
    num_iterations: int
    num_burn_in_iterations: int
