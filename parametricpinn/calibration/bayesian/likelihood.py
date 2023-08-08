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
    num_data = data.num_data_points
    size_flattened_outputs = data.num_data_points * data.dim_outputs

    def create_error_covariance_matrix(size_output: int, std_noise: float) -> Tensor:
        return torch.diag(torch.full((size_output,), std_noise**2, device=device))

    def calculate_inv_and_det(matrix: Tensor) -> tuple[Tensor, Tensor]:
        if matrix.size() == (1,):
            matrix = torch.unsqueeze(matrix, 1)
        inv_matrix = torch.inverse(matrix)
        det_matrix = torch.det(matrix)
        return inv_matrix, det_matrix

    def calculate_normalizing_constant(det_cov_error: Tensor) -> Tensor:
        return 1 / (
            torch.pow(
                (2 * torch.tensor(math.pi, device=device)),
                torch.tensor(size_flattened_outputs, device=device) / 2,
            )
            * torch.pow(det_cov_error, 1 / 2)
        )

    cov_error = create_error_covariance_matrix(size_flattened_outputs, data.std_noise)
    inv_cov_error, det_cov_error = calculate_inv_and_det(cov_error)
    normalizing_constant = calculate_normalizing_constant(det_cov_error)

    def likelihood_func(parameters: Tensor) -> Tensor:
        model_inputs = torch.concat(
            (
                inputs,
                parameters.repeat((num_data, 1)),
            ),
            dim=1,
        ).to(device)
        prediction = model(model_inputs)
        flattened_prediction = prediction.ravel()
        residual = flattened_prediction - flattened_outputs

        return normalizing_constant * (
            torch.exp(
                -1 / 2 * residual @ inv_cov_error @ torch.transpose(residual, -1, 0)
            )
        )

    return likelihood_func
