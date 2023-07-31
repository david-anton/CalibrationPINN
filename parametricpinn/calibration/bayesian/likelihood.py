import math
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.data import PreprocessedData
from parametricpinn.types import Device, Module, Tensor

LikelihoodFunc: TypeAlias = Callable[[Tensor], Tensor]


def compile_likelihood(
    model: Module,
    data: PreprocessedData,
    device: Device,
) -> LikelihoodFunc:
    def _calculate_inv_and_det_of_cov_matrix(
        cov_matrix: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if cov_matrix.size() == (1,):
            cov_matrix = torch.unsqueeze(cov_matrix, 1)
        inv_cov_matrix = torch.inverse(cov_matrix)
        det_cov_matrix = torch.det(cov_matrix)
        return inv_cov_matrix, det_cov_matrix

    inputs = data.inputs
    flattened_outputs = data.outputs.ravel()
    num_data_points = data.num_data_points
    covariance_error = data.error_covariance_matrix
    (
        inv_cov_error,
        det_cov_error,
    ) = _calculate_inv_and_det_of_cov_matrix(covariance_error)

    def likelihood_func(parameters: Tensor) -> Tensor:
        model_input = torch.concat(
            (
                inputs,
                parameters.repeat((num_data_points, 1)),
            ),
            dim=1,
        ).to(device)
        prediction = model(model_input).detach()
        flattened_prediction = prediction.ravel()
        residual = flattened_prediction - flattened_outputs
        return (
            1
            / (
                torch.pow(
                    (2 * torch.tensor(math.pi)), torch.tensor(num_data_points) / 2
                )
                * torch.pow(det_cov_error, 1 / 2)
            )
        ) * (
            torch.exp(
                -1 / 2 * residual @ inv_cov_error @ torch.transpose(residual, -1, 0)
            )
        )

    return likelihood_func
