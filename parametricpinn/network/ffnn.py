import math
from collections import OrderedDict
from typing import Callable, NamedTuple, TypeAlias, cast

import torch
import torch.nn as nn

from parametricpinn.types import Module, Tensor

InitializationFunc: TypeAlias = Callable[[Tensor], Tensor]
Layers: TypeAlias = list[Module]


class ParameterSet(NamedTuple):
    name: str
    num_parameters: int
    shape: torch.Size


ParameterStructure: TypeAlias = tuple[ParameterSet, ...]
FlattenedParameters: TypeAlias = Tensor
FlattenedGradients: TypeAlias = Tensor


class LinearHiddenLayer(nn.Module):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        activation: Module,
        init_weights: InitializationFunc,
        init_bias: InitializationFunc,
    ) -> None:
        super().__init__()
        self._fc_layer = nn.Linear(
            in_features=size_input,
            out_features=size_output,
            bias=True,
        )
        self._activation = activation
        init_weights(self._fc_layer.weight)
        init_bias(self._fc_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self._activation(self._fc_layer(x))


class LinearOutputLayer(nn.Module):
    def __init__(
        self,
        size_input: int,
        size_output: int,
        init_weights: InitializationFunc,
        init_bias: InitializationFunc,
    ) -> None:
        super().__init__()
        self._fc_layer = nn.Linear(
            in_features=size_input,
            out_features=size_output,
            bias=True,
        )
        init_weights(self._fc_layer.weight)
        init_bias(self._fc_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self._fc_layer(x)


class FFNN(Module):
    def __init__(
        self,
        layer_sizes: list[int],
        activation=nn.Tanh(),
        init_weights=nn.init.xavier_uniform_,
        init_bias=nn.init.zeros_,
    ) -> None:
        super().__init__()
        self._layers = self._set_up_layers(
            layer_sizes, activation, init_weights, init_bias
        )
        self._output = nn.Sequential(*self._layers)
        self._parameter_structure = self._determine_parameter_structure()

    def forward(self, x: Tensor) -> Tensor:
        return self._output(x)

    def get_flattened_parameters(self) -> FlattenedParameters:
        flattened_parameter_sets = []
        for parameter_set in self._output.parameters():
            flattened_parameter_sets.append(parameter_set.ravel())
        return torch.concat(flattened_parameter_sets, dim=0)

    def set_flattened_parameters(
        self, flattened_parameters: FlattenedParameters
    ) -> None:
        state_dict = self._create_state_dict(flattened_parameters)
        self._output.load_state_dict(state_dict=state_dict, strict=True)

    def get_flattened_gradients(self) -> FlattenedGradients:
        flattened_gradient_sets = []
        for parameter_set in self._output.parameters():
            grad_parameter_set = cast(torch.Tensor, parameter_set.grad)
            flattened_gradient_sets.append(grad_parameter_set.ravel())
        return torch.concat(flattened_gradient_sets, dim=0)

    def _set_up_layers(
        self,
        layer_sizes: list[int],
        activation: Module,
        init_weights: InitializationFunc,
        init_bias: InitializationFunc,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = [
            LinearHiddenLayer(
                size_input=layer_sizes[i - 1],
                size_output=layer_sizes[i],
                activation=activation,
                init_weights=init_weights,
                init_bias=init_bias,
            )
            for i in range(1, len(layer_sizes) - 1)
        ]

        layer_out = LinearOutputLayer(
            size_input=layer_sizes[-2],
            size_output=layer_sizes[-1],
            init_weights=init_weights,
            init_bias=init_bias,
        )
        layers.append(layer_out)
        return layers

    def _determine_parameter_structure(self) -> ParameterStructure:
        parameter_structure = []
        for name, parameters in self._output.named_parameters():
            num_parameters = torch.numel(parameters)
            shape = parameters.shape
            parameter_structure.append(
                ParameterSet(name=name, num_parameters=num_parameters, shape=shape)
            )
        return tuple(parameter_structure)

    def _create_state_dict(self, flattened_parameters) -> OrderedDict:
        state_dict = OrderedDict()
        start = 0
        for parameter_set in self._parameter_structure:
            end = start + parameter_set.num_parameters
            parameters = torch.reshape(
                flattened_parameters[start:end], parameter_set.shape
            )
            state_dict[parameter_set.name] = parameters
            start = end
        return state_dict


class SineFFNN(Module):
    def __init__(
        self,
        layer_sizes: list[int],
        activation=nn.Tanh(),
        init_weights=nn.init.xavier_uniform_,
        init_bias=nn.init.zeros_,
    ) -> None:
        super().__init__()
        self._layers = self._set_up_layers(
            layer_sizes, activation, init_weights, init_bias
        )
        self._output = nn.Sequential(*self._layers)
        self._parameter_structure = self._determine_parameter_structure()

    def forward(self, x: Tensor) -> Tensor:
        sine_x = torch.sin(
            torch.tensor(math.pi / 2, requires_grad=True, device=x.device) * x
        )
        return self._output(sine_x)

    def get_flattened_parameters(self) -> FlattenedParameters:
        flattened_parameter_sets = []
        for parameter_set in self._output.parameters():
            flattened_parameter_sets.append(parameter_set.ravel())
        return torch.concat(flattened_parameter_sets, dim=0)

    def set_flattened_parameters(
        self, flattened_parameters: FlattenedParameters
    ) -> None:
        state_dict = self._create_state_dict(flattened_parameters)
        self._output.load_state_dict(state_dict=state_dict, strict=True)

    def get_flattened_gradients(self) -> FlattenedGradients:
        flattened_gradient_sets = []
        for parameter_set in self._output.parameters():
            grad_parameter_set = cast(torch.Tensor, parameter_set.grad)
            flattened_gradient_sets.append(grad_parameter_set.ravel())
        return torch.concat(flattened_gradient_sets, dim=0)

    def _set_up_layers(
        self,
        layer_sizes: list[int],
        activation: Module,
        init_weights: InitializationFunc,
        init_bias: InitializationFunc,
    ) -> list[nn.Module]:
        layers: list[nn.Module] = [
            LinearHiddenLayer(
                size_input=layer_sizes[i - 1],
                size_output=layer_sizes[i],
                activation=activation,
                init_weights=init_weights,
                init_bias=init_bias,
            )
            for i in range(1, len(layer_sizes) - 1)
        ]

        layer_out = LinearOutputLayer(
            size_input=layer_sizes[-2],
            size_output=layer_sizes[-1],
            init_weights=init_weights,
            init_bias=init_bias,
        )
        layers.append(layer_out)
        return layers

    def _determine_parameter_structure(self) -> ParameterStructure:
        parameter_structure = []
        for name, parameters in self._output.named_parameters():
            num_parameters = torch.numel(parameters)
            shape = parameters.shape
            parameter_structure.append(
                ParameterSet(name=name, num_parameters=num_parameters, shape=shape)
            )
        return tuple(parameter_structure)

    def _create_state_dict(self, flattened_parameters) -> OrderedDict:
        state_dict = OrderedDict()
        start = 0
        for parameter_set in self._parameter_structure:
            end = start + parameter_set.num_parameters
            parameters = torch.reshape(
                flattened_parameters[start:end], parameter_set.shape
            )
            state_dict[parameter_set.name] = parameters
            start = end
        return state_dict
