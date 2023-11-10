from typing import Protocol, TypeAlias, Union

import torch
import torch.nn as nn

from parametricpinn.network import BFFNN, FFNN
from parametricpinn.types import Tensor

Networks: TypeAlias = Union[FFNN, BFFNN]
StandardNetworks: TypeAlias = FFNN
BayesianNetworks: TypeAlias = BFFNN
FlattenedParameters: TypeAlias = Tensor


class AnsatzStrategy(Protocol):
    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        pass


class StandardAnsatz(nn.Module):
    def __init__(
        self, network: StandardNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__()
        self.network = network
        self._ansatz_strategy = ansatz_strategy

    def forward(self, input: Tensor) -> Tensor:
        return self._ansatz_strategy(input, self.network)


class BayesianAnsatz(nn.Module):
    def __init__(
        self, network: BayesianNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__()
        self.network = network
        self._ansatz_strategy = ansatz_strategy

    def forward(self, input: Tensor) -> Tensor:
        return self._ansatz_strategy(input, self.network)

    def predict_normal_distribution(
        self, input: Tensor, parameter_samples: Tensor
    ) -> tuple[Tensor, Tensor]:
        prediction_list: list[Tensor] = []
        for parameter_sample in parameter_samples:
            self.network.set_flattened_parameters(parameter_sample)
            prediction = self.__call__(input)
            prediction_list.append(prediction)
        predictions = torch.stack(prediction_list, dim=0)
        means = torch.mean(predictions, dim=0)
        standard_deviations = torch.std(predictions, correction=0, dim=0)
        return means, standard_deviations


def extract_coordinate_1d(input: Tensor) -> Tensor:
    if input.dim() == 1:
        input_coor = torch.unsqueeze(input[0], 0)
    else:
        input_coor = input[:, 0].reshape((-1, 1))
    return input_coor


def extract_coordinates_2d(input: Tensor) -> tuple[Tensor, Tensor]:
    if input.dim() == 1:
        input_coor_x = torch.unsqueeze(input[0], dim=0)
        input_coor_y = torch.unsqueeze(input[1], dim=0)
    else:
        input_coor_x = input[:, 0].reshape((-1, 1))
        input_coor_y = input[:, 1].reshape((-1, 1))
    return input_coor_x, input_coor_y


def unbind_output(output: Tensor) -> tuple[Tensor, Tensor]:
    if output.dim() == 1:
        output_x, output_y = torch.unbind(output, dim=0)
        output_x = torch.unsqueeze(output_x, dim=0)
        output_y = torch.unsqueeze(output_y, dim=0)
    else:
        output_x, output_y = torch.unbind(output, dim=1)
        output_x = output_x.reshape((-1, 1))
        output_y = output_y.reshape((-1, 1))
    return output_x, output_y


def bind_output(output_x: Tensor, output_y: Tensor) -> Tensor:
    if output_x.dim() == 1 and output_y.dim() == 1:
        return torch.concat((output_x, output_y), dim=0)
    return torch.concat((output_x, output_y), dim=1)
