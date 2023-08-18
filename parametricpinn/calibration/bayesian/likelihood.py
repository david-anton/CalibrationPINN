import math
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.data import PreprocessedData
from parametricpinn.types import Device, Module, Tensor, TorchMultiNormalDist

LikelihoodFunc: TypeAlias = Callable[[Tensor], Tensor]


def compile_likelihood(
    model: Module,
    data: PreprocessedData,
    device: Device,
) -> LikelihoodFunc:
    inputs = data.inputs
    inputs.detach()
    flattened_outputs = data.outputs.ravel()
    size_data = data.num_data_points
    num_flattened_outputs = size_data * data.dim_outputs

    def create_error_covariance_matrix(num_outputs: int, std_noise: float) -> Tensor:
        return torch.diag(
            torch.full(
                (num_outputs,), std_noise**2, dtype=torch.float64, device=device
            )
        )

    def compile_likelihood(num_outputs: int, covariance_matrix) -> TorchMultiNormalDist:
        return torch.distributions.MultivariateNormal(
            loc=torch.zeros((num_outputs,), dtype=torch.float64, device=device),
            covariance_matrix=covariance_matrix,
        )

    cov_error = create_error_covariance_matrix(num_flattened_outputs, data.std_noise)
    likelihood = compile_likelihood(num_flattened_outputs, cov_error)

    def likelihood_func(parameters: Tensor) -> Tensor:
        model_inputs = torch.concat(
            (
                inputs,
                parameters.repeat((size_data, 1)),
            ),
            dim=1,
        ).to(device)
        prediction = model(model_inputs)
        flattened_prediction = prediction.ravel()
        residual = flattened_prediction - flattened_outputs

        return torch.exp(likelihood.log_prob(residual))

    return likelihood_func
