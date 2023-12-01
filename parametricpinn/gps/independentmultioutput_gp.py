from itertools import groupby
from typing import TypeAlias

import gpytorch
import torch

from parametricpinn.bayesian.prior import MultipliedPriors, Prior
from parametricpinn.errors import UnvalidGPMultivariateNormaError
from parametricpinn.gps.base import GaussianProcess
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Device, GPMultivariateNormal, Tensor

GPMultivariateNormalList: TypeAlias = list[GPMultivariateNormal]


class IndependentMultiOutputGPMultivariateNormal(
    gpytorch.distributions.MultivariateNormal
):
    def __init__(self, multivariate_normals: GPMultivariateNormalList, device: Device):
        self._validate_equal_size(multivariate_normals)
        self._multivariate_normals = multivariate_normals
        combined_mean = self._combine_means().to(device)
        combined_covariance_matrix = self._combine_covariance_matrices().to(device)
        super(IndependentMultiOutputGPMultivariateNormal).__init__(
            combined_mean, combined_covariance_matrix
        )
        self._device = device

    def _validate_equal_size(
        self, multivariate_normals: GPMultivariateNormalList
    ) -> None:
        normal_sizes = [normal.loc.size()[0] for normal in multivariate_normals]
        grouped_sizes = groupby(normal_sizes)
        is_only_one_group = next(grouped_sizes, True) and not next(grouped_sizes, False)
        if not is_only_one_group:
            raise UnvalidGPMultivariateNormaError(
                "It is expected that the independent multivariate normal distributions are of equal size."
            )

    def _combine_means(self) -> Tensor:
        means = [normal.mean.loc for normal in self._multivariate_normals]
        return torch.concat(means, dim=0)

    def _combine_covariance_matrices(self) -> Tensor:
        num_output_dims = self._determine_number_of_output_dimensions()
        single_output_length = self._determine_single_output_length()
        total_output_length = self._determine_total_output_length()
        combined_covar_matrix = torch.zeros((total_output_length, total_output_length))
        start_index = 0
        for i in range(num_output_dims - 1):
            index_slice = slice(start_index, start_index + single_output_length)
            covar_matrix_i = self._multivariate_normals[i].covariance_matrix
            combined_covar_matrix[index_slice, index_slice] = covar_matrix_i
            start_index += single_output_length
        covar_matrix_last = self._multivariate_normals[-1].covariance_matrix
        combined_covar_matrix[start_index:, start_index:] = covar_matrix_last
        return combined_covar_matrix

    def _determine_single_output_length(self) -> int:
        return self._multivariate_normals[0].loc.size()[0]

    def _determine_number_of_output_dimensions(self) -> int:
        return len(self._multivariate_normals)

    def _determine_total_output_length(self) -> int:
        output_lengths = [
            normal.mean.loc.size()[0] for normal in self._multivariate_normals
        ]
        return sum(output_lengths)


class IndependentMultiOutputGP(gpytorch.models.GP):
    def __init__(
        self,
        independent_gps: list[GaussianProcess],
        device: Device,
    ) -> None:
        super(IndependentMultiOutputGP, self).__init__()
        self._gps_list = independent_gps
        for gp in self._gps_list:
            gp.to(device)
        self.num_gps = len(self._gps_list)
        self.num_hyperparameters = self._determine_number_of_hyperparameetrs()
        self._device = device

    def forward(self, x) -> GPMultivariateNormalList:
        multivariate_normals = [gp.forward() for gp in self._gps_list]
        return IndependentMultiOutputGPMultivariateNormal(
            multivariate_normals, self._device
        )

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        num_outputs = self.num_gps
        num_inputs_1 = x_1.size()[0]
        num_inputs_2 = x_2.size()[0]
        covar_matrix = torch.zeros(
            (num_outputs * num_inputs_1, num_outputs * num_inputs_2),
            dtype=torch.float64,
            device=self._device,
        )
        start_index_1 = start_index_2 = 0
        for i in range(num_outputs - 1):
            index_slice_1 = slice(start_index_1, start_index_1 + num_inputs_1)
            index_slice_2 = slice(start_index_2, start_index_2 + num_inputs_2)
            covar_matrix_i = (
                self._gps_list[i].kernel(x_1, x_2).to_dense().to(self._device)
            )
            covar_matrix[index_slice_1, index_slice_2] = covar_matrix_i
            start_index_1 += num_inputs_1
            start_index_2 += num_inputs_2
        covar_matrix_last = (
            self._gps_list[-1].kernel(x_1, x_2).to_dense().to(self._device)
        )
        covar_matrix[start_index_1:, start_index_2:] = covar_matrix_last
        return covar_matrix

    def set_parameters(self, parameters: Tensor) -> None:
        self._validate_hyperparameters_size(parameters)
        start_index = 0
        for gp in self._gps_list[:-1]:
            num_parameters = gp.num_hyperparameters
            gp.set_parameters(
                parameters[start_index : start_index + num_parameters].to(self._device)
            )
            start_index += num_parameters
        gp_last = self._gps_list[-1]
        num_parameters = gp_last.num_hyperparameters
        gp_last.set_parameters(
            parameters[start_index : start_index + num_parameters].to(self._device)
        )

    def get_uninformed_parameters_prior(self, device: Device) -> Prior:
        priors = [gp.get_uninformed_parameters_prior(device) for gp in self._gps_list]
        return MultipliedPriors(priors)

    def _determine_number_of_hyperparameetrs(self) -> None:
        num_parameters_list = [gp.num_hyperparameters for gp in self._gps_list]
        return sum(num_parameters_list)

    def _validate_hyperparameters_size(self, parameters: Tensor) -> None:
        num_hyperparameters = sum([gp.num_hyperparameters for gp in self._gps_list])
        validate_parameters_size(parameters, torch.Size([num_hyperparameters]))

    def __call__(self, x: Tensor) -> GPMultivariateNormalList:
        return self.forward(x)
