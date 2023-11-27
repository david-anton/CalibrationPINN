from typing import Optional, TypeAlias
from itertools import groupby

import gpytorch
import torch

from parametricpinn.bayesian.prior import (
    Prior,
    create_mixed_independent_multivariate_distributed_prior,
)
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.statistics.distributions import (
    create_univariate_uniform_distribution,
)
from parametricpinn.types import Device, Tensor

from parametricpinn.gps.base import GaussianProcess
from parametricpinn.errors import UnvalidGPMultivariateNormaError

GPMultivariateNormal: TypeAlias = gpytorch.distributions.MultivariateNormal


class IndependentMultiOutputGPMultivariateNormal(
    gpytorch.distributions.MultivariateNormal
):
    def __init__(self, multivariate_normals: list[GPMultivariateNormal]):
        self._validate_equal_size(multivariate_normals)
        self._multivariate_normals = multivariate_normals
        combined_mean = self._combine_means()
        combined_covariance_matrix = self._combine_covariance_matrices()
        super(IndependentMultiOutputGPMultivariateNormal).__init__(
            mean=combined_mean, covariance_matrix=combined_covariance_matrix
        )

    def _validate_equal_size(
        self, multivariate_normals: list[GPMultivariateNormal]
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
        combined_covar_matrix = torch.zeros(total_output_length)
        start_index = 0
        for i in range(num_output_dims - 1):
            index_slice = slice(start_index, start_index + single_output_length)
            covar_matrix_i = self._multivariate_normals[i].covariance_matrix
            combined_covar_matrix[index_slice, index_slice] = covar_matrix_i
            start_index += single_output_length
        covar_matrix_last = self._multivariate_normals[num_output_dims].covariance_matrix
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


class IndependentMultiOutputGP:
    def __init__(
        self,
        gp_list: list[GaussianProcess],
        device: Device,
    ) -> None:
        for gp in self.gp_list:
            gp.to(device)
        self._gp_list = gpytorch.models.IndependentModelList(*gp_list)
        self.num_gps = len(gp_list)

    def forward(self, x) -> list[GPMultivariateNormal]:
        return self._gp_list.forward(x)

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        lazy_tensor = self.kernel(x_1, x_2)
        return lazy_tensor.evaluate()

    # def set_covariance_parameters(self, parameters: Tensor) -> None:
    #     valid_size = torch.Size([2])
    #     validate_parameters_size(parameters, valid_size)
    #     self.kernel.outputscale = parameters[0]
    #     self.kernel.base_kernel.lengthscale = parameters[1]

    # def get_uninformed_covariance_parameters_prior(self, device: Device) -> Prior:
    #     outputscale_prior = create_univariate_uniform_distribution(
    #         lower_limit=0.0, upper_limit=10.0, device=device
    #     )
    #     lengthscale_prior = create_univariate_uniform_distribution(
    #         lower_limit=0.0, upper_limit=10.0, device=device
    #     )
    #     return create_mixed_independent_multivariate_distributed_prior(
    #         [outputscale_prior, lengthscale_prior]
    #     )
