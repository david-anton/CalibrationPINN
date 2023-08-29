from typing import Callable, TypeAlias

import torch
import torch.nn as nn

from parametricpinn.types import Module, Tensor

InitializationFunc: TypeAlias = Callable[[Tensor], Tensor]
Layers: TypeAlias = list[Module]
ParameterSet: TypeAlias = tuple[str, int, torch.Size]
ParameterStructure: TypeAlias = tuple[ParameterSet, ...]


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
        self._layers = self._set_up_layers(layer_sizes, activation, init_weights, init_bias)
        self._output = nn.Sequential(*self._layers)
        self._parameter_structure = self._determine_parameter_structure()

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
        for name, parameter_set in self._output.named_parameters():
            num_parameters = torch.numel(parameter_set)
            shape = parameter_set.shape
            parameter_structure.append((name, num_parameters, shape))
        return tuple(parameter_structure)

    def forward(self, x: Tensor) -> Tensor:
        return self._output(x)
