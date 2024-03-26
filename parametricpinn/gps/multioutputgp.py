from itertools import groupby
from typing import TypeAlias

import gpytorch
import torch

from parametricpinn.errors import UnvalidGPMultivariateNormalError
from parametricpinn.gps.base import GPMultivariateNormal, NamedParameters
from parametricpinn.gps.gp import GP
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Device, Tensor

GPMultivariateNormalList: TypeAlias = list[GPMultivariateNormal]


class IndependentMultiOutputGPMultivariateNormal(GPMultivariateNormal):
    def __init__(self, multivariate_normals: GPMultivariateNormalList, device: Device):
        self._validate_equal_size(multivariate_normals)
        self._multivariate_normals = multivariate_normals
        combined_mean = self._combine_means().to(device)
        combined_covariance_matrix = self._combine_covariance_matrices().to(device)
        super().__init__(combined_mean, combined_covariance_matrix)
        self._device = device

    def _validate_equal_size(
        self, multivariate_normals: GPMultivariateNormalList
    ) -> None:
        normal_sizes = [normal.loc.size()[0] for normal in multivariate_normals]
        grouped_sizes = groupby(normal_sizes)
        is_only_one_group = next(grouped_sizes, True) and not next(grouped_sizes, False)
        if not is_only_one_group:
            raise UnvalidGPMultivariateNormalError(
                "It is expected that the independent multivariate normal distributions are of equal size."
            )

    def _combine_means(self) -> Tensor:
        means = tuple(normal.mean.loc for normal in self._multivariate_normals)
        return torch.concat(means, dim=0)

    def _combine_covariance_matrices(self) -> Tensor:
        output_dims = self._determine_number_of_output_dimensions()
        output_size = self._determine_output_size()

        sub_covar_matrices = []
        sub_covar_matrices.append(
            torch.concat(
                (
                    self._multivariate_normals[0].covariance_matrix,
                    _create_zeros(
                        output_size, (output_dims - 1) * output_size, self._device
                    ),
                ),
                dim=1,
            )
        )

        for i in range(1, output_dims - 1):
            zeros_left = _create_zeros(output_size, i * output_size, self._device)
            sub_covar_matrix = self._multivariate_normals[i].covariance_matrix
            zeros_right = _create_zeros(
                output_size, (output_dims - 1 - i) * output_size, self._device
            )
            sub_covar_matrices.append(
                torch.concat((zeros_left, sub_covar_matrix, zeros_right), dim=1)
            )

        sub_covar_matrices.append(
            torch.concat(
                (
                    _create_zeros(
                        output_size, (output_dims - 1) * output_size, self._device
                    ),
                    self._multivariate_normals[-1].covariance_matrix,
                ),
                dim=1,
            )
        )
        return torch.concat(sub_covar_matrices, dim=0)

    def _determine_number_of_output_dimensions(self) -> int:
        return len(self._multivariate_normals)

    def _determine_output_size(self) -> int:
        return self._multivariate_normals[0].loc.size()[0]


class IndependentMultiOutputGP(gpytorch.models.GP):
    def __init__(
        self,
        gps: list[GP],
        device: Device,
    ) -> None:
        super().__init__()
        for gp in gps:
            gp.to(device)
        self._gps = torch.nn.ModuleList(gps)
        self.num_gps = len(self._gps)
        self.num_hyperparameters = self._determine_number_of_hyperparameters()
        self._device = device

    def forward(self, x: Tensor) -> GPMultivariateNormal:
        multivariate_normals = [gp.forward(x) for gp in self._gps]
        return IndependentMultiOutputGPMultivariateNormal(
            multivariate_normals, self._device
        )

    def forward_mean(self, x: Tensor) -> Tensor:
        means = [gp.forward_mean(x) for gp in self._gps]
        return torch.concat(means, dim=0).to(self._device)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        num_outputs = self.num_gps
        num_inputs_1 = x_1.size()[0]
        num_inputs_2 = x_2.size()[0]

        sub_covar_matrices = []
        sub_covar_matrices.append(
            torch.concat(
                (
                    self._gps[0].forward_kernel(x_1, x_2),
                    _create_zeros(
                        num_inputs_1, (num_outputs - 1) * num_inputs_2, self._device
                    ),
                ),
                dim=1,
            )
        )

        for i in range(1, num_outputs - 1):
            zeros_left = _create_zeros(num_inputs_1, i * num_inputs_2, self._device)
            sub_covar_matrix = self._gps[i].forward_kernel(x_1, x_2)
            zeros_right = _create_zeros(
                num_inputs_1, (num_outputs - 1 - i) * num_inputs_2, self._device
            )
            sub_covar_matrices.append(
                torch.concat((zeros_left, sub_covar_matrix, zeros_right), dim=1)
            )

        sub_covar_matrices.append(
            torch.concat(
                (
                    _create_zeros(
                        num_inputs_1, (num_outputs - 1) * num_inputs_2, self._device
                    ),
                    self._gps[-1].forward_kernel(x_1, x_2),
                ),
                dim=1,
            )
        )
        return torch.concat(sub_covar_matrices, dim=0)

    def set_parameters(self, parameters: Tensor) -> None:
        validate_parameters_size(parameters, self.num_hyperparameters)
        start_index = 0
        for gp in self._gps[:-1]:
            num_parameters = gp.num_hyperparameters
            gp.set_parameters(
                parameters[start_index : start_index + num_parameters].to(self._device)
            )
            start_index += num_parameters
        gp_last = self._gps[-1]
        num_parameters = gp_last.num_hyperparameters
        gp_last.set_parameters(
            parameters[start_index : start_index + num_parameters].to(self._device)
        )

    def get_named_parameters(self) -> NamedParameters:
        return {
            f"{key}_dim_{count}": value
            for count, gp in enumerate(self._gps)
            for key, value in gp.get_named_parameters().items()
        }

    def _determine_number_of_hyperparameters(self) -> int:
        return sum([gp.num_hyperparameters for gp in self._gps])


def _create_zeros(dim_1: int, dim_2: int, device: Device) -> Tensor:
    return torch.zeros(
        (dim_1, dim_2),
        dtype=torch.float64,
        requires_grad=True,
        device=device,
    )
