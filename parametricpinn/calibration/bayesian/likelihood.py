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
    inputs = data.inputs
    flattened_outputs = data.outputs.ravel()
    size_outputs = data.num_data_points
    size_flattened_outputs = data.num_data_points * data.dim_outputs

    def _create_error_covariance_matrix(size_output: int, std_noise: float) -> Tensor:
        return torch.diag(torch.full((size_output,), std_noise**2))

    def _calculate_inv_and_det(covariance_matrix: Tensor) -> tuple[Tensor, Tensor]:
        if covariance_matrix.size() == (1,):
            covariance_matrix = torch.unsqueeze(covariance_matrix, 1)
        inv_cov_matrix = torch.inverse(covariance_matrix)
        det_cov_matrix = torch.det(covariance_matrix)
        return inv_cov_matrix, det_cov_matrix

    covariance_error = _create_error_covariance_matrix(
        size_output=size_flattened_outputs, std_noise=data.std_noise
    )
    inv_cov_error, det_cov_error = _calculate_inv_and_det(covariance_error)

    def likelihood_func(parameters: Tensor) -> Tensor:
        model_inputs = torch.concat(
            (
                inputs,
                parameters.repeat((size_outputs, 1)),
            ),
            dim=1,
        ).to(device)
        prediction = model(model_inputs).detach()
        flattened_prediction = prediction.ravel()
        residual = flattened_prediction - flattened_outputs

        return (
            1
            / (
                torch.pow(
                    (2 * torch.tensor(math.pi)),
                    torch.tensor(size_flattened_outputs) / 2,
                )
                * torch.pow(det_cov_error, 1 / 2)
            )
        ) * (
            torch.exp(
                -1 / 2 * residual @ inv_cov_error @ torch.transpose(residual, -1, 0)
            )
        )

    return likelihood_func
