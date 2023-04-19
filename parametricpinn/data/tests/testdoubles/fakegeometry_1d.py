# Standard library imports

# Third-party imports
import torch

# Local library imports
from parametricpinn.types import Tensor


class FakeGeometry1D:
    def __init__(self, length) -> None:
        self.length = length

    def create_uniform_points(self, num_points: int) -> Tensor:
        return torch.tensor([[0.0], [10.0]])

    def create_random_points(self, num_points: int) -> Tensor:
        return torch.tensor([[0.0], [5.0], [10.0]])

    def create_point_at_free_end(self) -> Tensor:
        return torch.tensor([[10.0]])
