from typing import NamedTuple

import torch
import torch.nn as nn

from parametricpinn.bayesian.prior import (
    Prior,
    create_independent_multivariate_normal_distributed_prior,
)
from parametricpinn.errors import BayesianNNError
from parametricpinn.network.ffnn import FFNN, ParameterSet
from parametricpinn.types import Device, Tensor


class ParameterPriorStds(NamedTuple):
    weight: float
    bias: float


class BFFNN(FFNN):
    def __init__(self, layer_sizes: list[int]) -> None:
        super().__init__(
            layer_sizes,
            activation=nn.Tanh(),
            init_weights=nn.init.zeros_,
            init_bias=nn.init.zeros_,
        )

    def create_independent_multivariate_normal_prior(
        self, parameter_stds: ParameterPriorStds, device: Device
    ) -> Prior:
        means = self._create_means()
        standard_deviations = self._assemble_standard_deviations(parameter_stds)
        return create_independent_multivariate_normal_distributed_prior(
            means, standard_deviations, device
        )

    def _create_means(self) -> Tensor:
        means = []
        for parameter_set in self._parameter_structure:
            set_means = torch.zeros((parameter_set.num_parameters,))
            means.append(set_means)
        return torch.concat(means, dim=0)

    def _assemble_standard_deviations(
        self, parameter_stds: ParameterPriorStds
    ) -> Tensor:
        standard_deviations = []
        for parameter_set in self._parameter_structure:
            one_standard_deviation = self._assign_standard_deviation(
                parameter_set, parameter_stds
            )
            set_standard_deviations = torch.full(
                (parameter_set.num_parameters,), one_standard_deviation
            )
            standard_deviations.append(set_standard_deviations)
        return torch.concat(standard_deviations, dim=0)

    def _assign_standard_deviation(
        self, parameter_set: ParameterSet, parameter_stds: ParameterPriorStds
    ) -> float:
        parameter_name = parameter_set.name.split(sep=".")[-1]
        parameter_stds_dict = parameter_stds._asdict()
        if parameter_name in list(parameter_stds_dict.keys()):
            return parameter_stds_dict[parameter_name]
        else:
            raise BayesianNNError(f"Parameter {parameter_name} not supported.")
