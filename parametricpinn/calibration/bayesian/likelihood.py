from typing import Callable, TypeAlias

import numpy as np
import torch

from parametricpinn.calibration.utility import _freeze_model
from parametricpinn.errors import InputDataDoesNotMatchError
from parametricpinn.types import Device, Module, NPArray

LikelihoodFunc: TypeAlias = Callable[[NPArray], float]


def compile_likelihood(
    model: Module,
    coordinates: NPArray,
    data: NPArray,
    covariance_error: NPArray,
    device: Device,
) -> LikelihoodFunc:
    def _prepare_model(model: Module) -> None:
        model.to(device)
        _freeze_model(model)

    def _validate_data(data: NPArray) -> None:
        if not coordinates.shape[0] == data.shape[0]:
            raise InputDataDoesNotMatchError(
                "Shape of coordinates and data for calibration does not match!"
            )

    def _calculate_inverse_and_determinant_of_covariance_matrix(
        cov_matrix: NPArray,
    ) -> tuple[NPArray, NPArray]:
        if cov_matrix.shape == (1,):
            cov_matrix = np.array([cov_matrix])
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        det_cov_matrix = np.linalg.det(cov_matrix)
        return inv_cov_matrix, det_cov_matrix

    _prepare_model(model)
    _validate_data(data)
    dim_data = data.shape[0]
    (
        inv_cov_error,
        det_cov_error,
    ) = _calculate_inverse_and_determinant_of_covariance_matrix(covariance_error)

    def likelihood_func(parameters: NPArray) -> float:
        model_input = torch.concat(
            (
                torch.from_numpy(coordinates),
                torch.from_numpy(parameters).repeat((dim_data, 1)),
            ),
            dim=1,
        ).to(device)
        prediction = model(model_input).detach().cpu().numpy()
        residual = (prediction - data).ravel()
        return (
            1 / (np.power(2 * np.pi, dim_data / 2) * np.power(det_cov_error, 1 / 2))
        ) * (np.exp(-1 / 2 * residual @ inv_cov_error @ residual.T))

    return likelihood_func
