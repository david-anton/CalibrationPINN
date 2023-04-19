# Standard library imports
from typing import Protocol

# Third-party imports

# Local library imports
from parametricpinn.types import Tensor


class Geometry1DProtocol(Protocol):
    def create_uniform_points(self, num_points: int) -> Tensor:
        "Returns uniformly distributed points within the geometry."

    def create_random_points(self, num_points: int) -> Tensor:
        "Returns randomly distributed points within the geometry."

    def create_point_at_free_end(self) -> Tensor:
        "Returns one point at the free end."


class Geometry2DProtocol(Protocol):
    def create_random_points(self, num_points: int) -> Tensor:
        "Returns randomly distributed points within the geometry."

    def create_uniform_points_on_left_boundary(self, num_points):
        "Returns uniformly distributed points on left boundary."
