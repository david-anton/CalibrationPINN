from parametricpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
    extract_coordinate_1d,
)
from parametricpinn.ansatz.distancefunctions import (
    DistanceFunction,
    distance_function_factory,
)
from parametricpinn.types import Tensor


class HBCAnsatzStrategyStretchedRod:
    def __init__(
        self, displacement_left: Tensor, distance_func: DistanceFunction
    ) -> None:
        super().__init__()
        self._displacement_left = displacement_left
        self._distance_func = distance_func

    def _boundary_data_func(self) -> Tensor:
        return self._displacement_left

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        input_coor = extract_coordinate_1d(input)
        return self._boundary_data_func() + (
            self._distance_func(input_coor) * network(input)
        )


def create_standard_hbc_ansatz_stretched_rod(
    displacement_left: Tensor,
    range_coordinate: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
) -> StandardAnsatz:
    distance_func = distance_function_factory(distance_function_type, range_coordinate)
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(displacement_left, distance_func)
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_hbc_ansatz_stretched_rod(
    displacement_left: Tensor,
    range_coordinate: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
) -> BayesianAnsatz:
    distance_func = distance_function_factory(distance_function_type, range_coordinate)
    ansatz_strategy = HBCAnsatzStrategyStretchedRod(displacement_left, distance_func)
    return BayesianAnsatz(network, ansatz_strategy)
