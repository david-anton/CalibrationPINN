import torch

from parametricpinn.types import Tensor


class StretchedRod1D:
    def __init__(
        self,
        length: float,
    ) -> None:
        self.length = length
        self._sobol_engine = torch.quasirandom.SobolEngine(dimension=1)

    def create_uniform_points(self, num_points: int) -> Tensor:
        return torch.linspace(0.0, self.length, num_points, requires_grad=True).view(
            num_points, 1
        )

    def create_quasi_random_sobol_points(self, num_points: int) -> Tensor:
        min_coordinate = torch.tensor([0.0])
        normalized_lengths = self._sobol_engine.draw(num_points)
        length = normalized_lengths * self.length
        return min_coordinate + length
    
    def create_random_points(self, num_points: int) -> Tensor:
        return torch.rand((num_points, 1), requires_grad=True) * self.length

    def create_point_at_free_end(self) -> Tensor:
        return torch.full((1, 1), self.length, requires_grad=True)
