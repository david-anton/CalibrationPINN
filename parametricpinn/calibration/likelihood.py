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
    model.to(device)
    _freeze_model(model)
    if not coordinates.shape[0] == data.shape[0]:
        raise InputDataDoesNotMatchError(
            "Shape of coordinates and data for calibration does not match!"
        )
    dim_data = data.shape[0]
    if covariance_error.shape == (1,):
        covariance_error_2D = np.array([covariance_error])
        inverse_covariance_error = np.linalg.inv(covariance_error_2D)[0]
        determinant_covariance_error = np.linalg.det(covariance_error_2D)
    else:
        inverse_covariance_error = np.linalg.inv(covariance_error)
        determinant_covariance_error = np.linalg.det(covariance_error)

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
            1
            / (
                np.power(2 * np.pi, dim_data / 2)
                * np.power(determinant_covariance_error, 1 / 2)
            )
        ) * (np.exp(-1 / 2 * residual.T * inverse_covariance_error * residual))[0]

    return likelihood_func
