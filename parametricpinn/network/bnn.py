from typing import TypeAlias

import torch
import torch.nn as nn

from parametricpinn.network.ffnn import FFNN
from parametricpinn.types import Tensor

FlattenedParameters: TypeAlias = Tensor


class BNN(FFNN):
    def __init__(self, layer_sizes: list[int]) -> None:
        super().__init__(
            layer_sizes,
            activation=nn.Tanh(),
            init_weights=nn.init.zeros_,
            init_bias=nn.init.zeros_,
        )

    def get_flattened_parameters(self) -> FlattenedParameters:
        flattened_parameter_sets = []
        for _, parameter_set in self._output.named_parameters():
            flattened_parameter_sets.append(parameter_set.ravel())
        return torch.concat(flattened_parameter_sets, dim=0)
    
    def set_flattened_parameters(self, flattened_parameters: FlattenedParameters) -> None:
        pass
            
