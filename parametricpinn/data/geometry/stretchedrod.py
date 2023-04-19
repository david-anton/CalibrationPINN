# Standard library imports

# Third-party imports
import torch

# Local library imports
from parametricpinn.types import Tensor


class StretchedRod:
    def __init__(
        self,
        length: float,
    ) -> None:
        self.length = length

    def create_uniform_points(self, num_points: int) -> Tensor:
        return torch.linspace(0.0, self.length, num_points, requires_grad=True).view(
            num_points, 1
        )

    def create_random_points(self, num_points: int) -> Tensor:
        return torch.rand((num_points, 1)) * self.length

    def create_points_at_free_end(self, num_points: int) -> Tensor:
        return torch.full((num_points, 1), self.length, requires_grad=True)
