# from typing import TypeAlias

# import torch

# from parametricpinn.ansatz.base import (
#     BayesianAnsatz,
#     BayesianNetworks,
#     Networks,
#     StandardAnsatz,
#     StandardNetworks,
#     bind_output,
#     extract_coordinates_2d,
#     unbind_output,
# )
# from parametricpinn.ansatz.distancefunctions import (
#     DistanceFunction,
#     distance_function_factory,
# )
# from parametricpinn.ansatz.hbc_normalizers import (
#     HBCAnsatzNormalizer,
#     HBCAnsatzRenormalizer,
# )
# from parametricpinn.network.normalizednetwork import InputNormalizer
# from parametricpinn.types import Device, Tensor

# NetworkInputNormalizer: TypeAlias = InputNormalizer


# class NormalizedHBCAnsatzStrategyClampedLeft:
#     def __init__(
#         self,
#         network_input_normalizer: NetworkInputNormalizer,
#         hbc_ansatz_output_normalizer_x: HBCAnsatzNormalizer,
#         hbc_ansatz_output_normalizer_y: HBCAnsatzNormalizer,
#         distance_func_x: DistanceFunction,
#         hbc_ansatz_output_renormalizer: HBCAnsatzRenormalizer,
#         device: Device,
#     ) -> None:
#         super().__init__()
#         self._boundary_data_x_left = torch.tensor([0.0], device=device)
#         self._boundary_data_y_left = torch.tensor([0.0], device=device)
#         self._network_input_normalizer = network_input_normalizer
#         self._hbc_ansatz_output_normalizer_x = hbc_ansatz_output_normalizer_x
#         self._hbc_ansatz_output_normalizer_y = hbc_ansatz_output_normalizer_y
#         self._distance_func_x = distance_func_x
#         self._hbc_ansatz_output_renormalizer = hbc_ansatz_output_renormalizer

#     def _boundary_data_func_x(self) -> Tensor:
#         return self._hbc_ansatz_output_normalizer_x(self._boundary_data_x_left)

#     def _boundary_data_func_y(self) -> Tensor:
#         return self._hbc_ansatz_output_normalizer_y(self._boundary_data_y_left)

#     def __call__(self, input: Tensor, network: Networks) -> Tensor:
#         input_coor_x, _ = extract_coordinates_2d(input)
#         norm_input = self._network_input_normalizer(input)
#         norm_network_output = network(norm_input)
#         norm_network_output_x, norm_network_output_y = unbind_output(
#             norm_network_output
#         )
#         norm_output_x = (
#             self._boundary_data_func_x()
#             + self._distance_func_x(input_coor_x) * norm_network_output_x
#         )
#         norm_output_y = (
#             self._boundary_data_func_y()
#             + self._distance_func_x(input_coor_x) * norm_network_output_y
#         )
#         norm_output = bind_output(norm_output_x, norm_output_y)
#         return self._hbc_ansatz_output_renormalizer(norm_output)


# def create_standard_normalized_hbc_ansatz_clamped_left(
#     coordinate_x_left: Tensor,
#     min_inputs: Tensor,
#     max_inputs: Tensor,
#     min_outputs: Tensor,
#     max_outputs: Tensor,
#     network: StandardNetworks,
#     distance_function_type: str,
#     device: Device,
# ) -> StandardAnsatz:
#     ansatz_strategy = _create_ansatz_strategy(
#         coordinate_x_left,
#         min_inputs,
#         max_inputs,
#         min_outputs,
#         max_outputs,
#         distance_function_type,
#         device,
#     )
#     return StandardAnsatz(network, ansatz_strategy)


# def create_bayesian_normalized_hbc_ansatz_clamped_left(
#     coordinate_x_left: Tensor,
#     min_inputs: Tensor,
#     max_inputs: Tensor,
#     min_outputs: Tensor,
#     max_outputs: Tensor,
#     network: BayesianNetworks,
#     distance_function_type: str,
#     device: Device,
# ) -> BayesianAnsatz:
#     ansatz_strategy = _create_ansatz_strategy(
#         coordinate_x_left,
#         min_inputs,
#         max_inputs,
#         min_outputs,
#         max_outputs,
#         distance_function_type,
#         device,
#     )
#     return BayesianAnsatz(network, ansatz_strategy)


# def _create_ansatz_strategy(
#     coordinate_x_left: Tensor,
#     min_inputs: Tensor,
#     max_inputs: Tensor,
#     min_outputs: Tensor,
#     max_outputs: Tensor,
#     distance_func_type: str,
#     device: Device,
# ) -> NormalizedHBCAnsatzStrategyClampedLeft:
#     network_input_normalizer = _create_network_input_normalizer(min_inputs, max_inputs)
#     (
#         ansatz_output_normalizer_x,
#         ansatz_output_normalizer_y,
#     ) = _create_ansatz_output_normalizers(min_outputs, max_outputs)
#     distance_func_x = _create_distance_function_x(
#         distance_func_type, min_inputs, max_inputs, coordinate_x_left
#     )
#     ansatz_output_renormalizer = _create_ansatz_output_renormalizer(
#         min_outputs, max_outputs
#     )
#     return NormalizedHBCAnsatzStrategyClampedLeft(
#         network_input_normalizer=network_input_normalizer,
#         hbc_ansatz_output_normalizer_x=ansatz_output_normalizer_x,
#         hbc_ansatz_output_normalizer_y=ansatz_output_normalizer_y,
#         distance_func_x=distance_func_x,
#         hbc_ansatz_output_renormalizer=ansatz_output_renormalizer,
#         device=device,
#     )


# def _create_network_input_normalizer(
#     min_inputs: Tensor, max_inputs: Tensor
# ) -> NetworkInputNormalizer:
#     return NetworkInputNormalizer(min_inputs, max_inputs)


# def _create_ansatz_output_normalizers(
#     min_outputs: Tensor, max_outputs: Tensor
# ) -> tuple[HBCAnsatzNormalizer, HBCAnsatzNormalizer]:
#     output_normalizer_x = _create_one_ansatz_output_normalizer(
#         min_outputs, max_outputs, index=0
#     )
#     output_normalizer_y = _create_one_ansatz_output_normalizer(
#         min_outputs, max_outputs, index=1
#     )
#     return output_normalizer_x, output_normalizer_y


# def _create_one_ansatz_output_normalizer(
#     min_outputs: Tensor, max_outputs: Tensor, index: int
# ) -> HBCAnsatzNormalizer:
#     return HBCAnsatzNormalizer(
#         torch.unsqueeze(min_outputs[index], dim=0),
#         torch.unsqueeze(max_outputs[index], dim=0),
#     )


# def _create_distance_function_x(
#     distance_func_type: str,
#     min_inputs: Tensor,
#     max_inputs: Tensor,
#     coordinate_x_left: Tensor,
# ) -> DistanceFunction:
#     idx_coordinate = 0
#     range_coordinate_x = torch.unsqueeze(
#         max_inputs[idx_coordinate] - min_inputs[idx_coordinate], dim=0
#     )
#     device = range_coordinate_x.device
#     boundary_coordinate_x = coordinate_x_left.to(device)
#     return distance_function_factory(
#         distance_func_type, range_coordinate_x, boundary_coordinate_x
#     )


# def _create_ansatz_output_renormalizer(
#     min_outputs: Tensor, max_outputs: Tensor
# ) -> HBCAnsatzRenormalizer:
#     return HBCAnsatzRenormalizer(min_outputs, max_outputs)


from typing import TypeAlias

import torch

from parametricpinn.ansatz.base import (
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
    bind_output,
    extract_coordinates_2d,
    unbind_output,
)
from parametricpinn.ansatz.distancefunctions import (
    DistanceFunction,
    distance_function_factory,
)
from parametricpinn.ansatz.hbc_normalizers import (
    HBCAnsatzNormalizer,
    HBCAnsatzRenormalizer,
)
from parametricpinn.network.normalizednetwork import InputNormalizer, OutputRenormalizer
from parametricpinn.types import Device, Tensor

NetworkInputNormalizer: TypeAlias = InputNormalizer
AnsatzOutputNormalizer: TypeAlias = InputNormalizer
AnsatzOutputRenormalizer: TypeAlias = OutputRenormalizer


class NormalizedHBCAnsatzStrategyClampedLeft:
    def __init__(
        self,
        network_input_normalizer: NetworkInputNormalizer,
        ansatz_output_normalizer_x: AnsatzOutputNormalizer,
        ansatz_output_normalizer_y: AnsatzOutputNormalizer,
        distance_func_x: DistanceFunction,
        ansatz_output_renormalizer: AnsatzOutputRenormalizer,
        device: Device,
    ) -> None:
        super().__init__()
        self._boundary_data_x_left = torch.tensor([0.0], device=device)
        self._boundary_data_y_left = torch.tensor([0.0], device=device)
        self._network_input_normalizer = network_input_normalizer
        self._ansatz_output_normalizer_x = ansatz_output_normalizer_x
        self._ansatz_output_normalizer_y = ansatz_output_normalizer_y
        self._distance_func_x = distance_func_x
        self._ansatz_output_renormalizer = ansatz_output_renormalizer

    def _boundary_data_func_x(self) -> Tensor:
        return self._ansatz_output_normalizer_x(self._boundary_data_x_left)

    def _boundary_data_func_y(self) -> Tensor:
        return self._ansatz_output_normalizer_y(self._boundary_data_y_left)

    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        coor_x, _ = extract_coordinates_2d(input)
        norm_netwotk_input = self._network_input_normalizer(input)
        norm_network_output = network(norm_netwotk_input)
        norm_network_output_x, norm_network_output_y = unbind_output(
            norm_network_output
        )
        norm_ansatz_output_x = (
            self._boundary_data_func_x()
            + self._distance_func_x(coor_x) * norm_network_output_x
        )
        norm_ansatz_output_y = (
            self._boundary_data_func_y()
            + self._distance_func_x(coor_x) * norm_network_output_y
        )
        norm_ansatz_output = bind_output(norm_ansatz_output_x, norm_ansatz_output_y)
        return self._ansatz_output_renormalizer(norm_ansatz_output)


def create_standard_normalized_hbc_ansatz_clamped_left(
    coordinate_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: StandardNetworks,
    distance_function_type: str,
    device: Device,
) -> StandardAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        coordinate_x_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
        device,
    )
    return StandardAnsatz(network, ansatz_strategy)


def create_bayesian_normalized_hbc_ansatz_clamped_left(
    coordinate_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    network: BayesianNetworks,
    distance_function_type: str,
    device: Device,
) -> BayesianAnsatz:
    ansatz_strategy = _create_ansatz_strategy(
        coordinate_x_left,
        min_inputs,
        max_inputs,
        min_outputs,
        max_outputs,
        distance_function_type,
        device,
    )
    return BayesianAnsatz(network, ansatz_strategy)


def _create_ansatz_strategy(
    coordinate_x_left: Tensor,
    min_inputs: Tensor,
    max_inputs: Tensor,
    min_outputs: Tensor,
    max_outputs: Tensor,
    distance_func_type: str,
    device: Device,
) -> NormalizedHBCAnsatzStrategyClampedLeft:
    network_input_normalizer = _create_network_input_normalizer(min_inputs, max_inputs)
    distance_func_x = _create_distance_function_x(
        distance_func_type, min_inputs, max_inputs, coordinate_x_left
    )
    (
        ansatz_output_normalizer_x,
        ansatz_output_normalizer_y,
    ) = _create_ansatz_output_normalizers(min_outputs, max_outputs)
    ansatz_output_renormalizer = _create_ansatz_output_renormalizer(
        min_outputs, max_outputs
    )
    return NormalizedHBCAnsatzStrategyClampedLeft(
        network_input_normalizer=network_input_normalizer,
        ansatz_output_normalizer_x=ansatz_output_normalizer_x,
        ansatz_output_normalizer_y=ansatz_output_normalizer_y,
        distance_func_x=distance_func_x,
        ansatz_output_renormalizer=ansatz_output_renormalizer,
        device=device,
    )


def _create_network_input_normalizer(
    min_inputs: Tensor, max_inputs: Tensor
) -> NetworkInputNormalizer:
    return NetworkInputNormalizer(min_inputs, max_inputs)


def _create_distance_function_x(
    distance_func_type: str,
    min_inputs: Tensor,
    max_inputs: Tensor,
    coordinate_x_left: Tensor,
) -> DistanceFunction:
    idx_coordinate = 0
    range_coordinate_x = torch.unsqueeze(
        max_inputs[idx_coordinate] - min_inputs[idx_coordinate], dim=0
    )
    device = range_coordinate_x.device
    boundary_coordinate_x = coordinate_x_left.to(device)
    return distance_function_factory(
        distance_func_type, range_coordinate_x, boundary_coordinate_x
    )


def _create_ansatz_output_normalizers(
    min_outputs: Tensor, max_outputs: Tensor
) -> AnsatzOutputNormalizer:
    ansatz_output_normalizer_x = _create_one_ansatz_output_normalizer(
        min_outputs, max_outputs, index=0
    )
    ansatz_output_normalizer_y = _create_one_ansatz_output_normalizer(
        min_outputs, max_outputs, index=1
    )
    return ansatz_output_normalizer_x, ansatz_output_normalizer_y


def _create_one_ansatz_output_normalizer(
    min_outputs: Tensor, max_outputs: Tensor, index: int
) -> AnsatzOutputNormalizer:
    return AnsatzOutputNormalizer(
        torch.unsqueeze(min_outputs[index], dim=0),
        torch.unsqueeze(max_outputs[index], dim=0),
    )


def _create_ansatz_output_renormalizer(
    min_outputs: Tensor, max_outputs: Tensor
) -> AnsatzOutputRenormalizer:
    return AnsatzOutputRenormalizer(min_outputs, max_outputs)
