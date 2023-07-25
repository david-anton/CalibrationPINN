import math
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.utility import _freeze_model
from parametricpinn.errors import CalibrationDataDoesNotMatchError
from parametricpinn.types import Device, Module, Tensor

LikelihoodFunc: TypeAlias = Callable[[Tensor], Tensor]


def compile_likelihood(
    model: Module,
    coordinates: Tensor,
    data: Tensor,
    covariance_error: Tensor,
    device: Device,
) -> LikelihoodFunc:
    def _prepare_model(model: Module) -> None:
        model.to(device)
        _freeze_model(model)

    def _validate_data(data: Tensor, coordinates: Tensor) -> None:
        if not data.size()[0] == coordinates.size()[0]:
            raise CalibrationDataDoesNotMatchError(
                "Shape of input and output for calibration does not match!"
            )

    def _calculate_inv_and_det_of_cov_matrix(
        cov_matrix: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if cov_matrix.size() == (1,):
            cov_matrix = torch.unsqueeze(cov_matrix, 1)
        inv_cov_matrix = torch.inverse(cov_matrix)
        det_cov_matrix = torch.det(cov_matrix)
        return inv_cov_matrix, det_cov_matrix

    _prepare_model(model)
    _validate_data(data, coordinates)
    dim_data = data.size()[0]
    (
        inv_cov_error,
        det_cov_error,
    ) = _calculate_inv_and_det_of_cov_matrix(covariance_error)

    def likelihood_func(parameters: Tensor) -> Tensor:
        model_input = torch.concat(
            (
                coordinates,
                parameters.repeat((dim_data, 1)),
            ),
            dim=1,
        ).to(device)
        prediction = model(model_input).detach()
        residual = (prediction - data).ravel()
        return (
            1
            / (
                torch.pow((2 * torch.tensor(math.pi)), torch.tensor(dim_data) / 2)
                * torch.pow(det_cov_error, 1 / 2)
            )
        ) * (
            torch.exp(
                -1 / 2 * residual @ inv_cov_error @ torch.transpose(residual, -1, 0)
            )
        )

    return likelihood_func
