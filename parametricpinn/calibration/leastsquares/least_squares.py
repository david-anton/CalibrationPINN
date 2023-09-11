from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.calibration.base import (
    CalibrationData,
    Parameters,
    preprocess_calibration_data,
)
from parametricpinn.calibration.config import CalibrationConfig
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.types import Device, NPArray, Tensor

LeastSquaresOutput: TypeAlias = tuple[NPArray, list[float]]
LeastSquaresFunc: TypeAlias = Callable[
    [
        StandardAnsatz,
        CalibrationData,
        Parameters,
        int,
        Device,
    ],
    LeastSquaresOutput,
]


@dataclass
class LeastSquaresConfig(CalibrationConfig):
    ansatz: StandardAnsatz
    calibration_data: CalibrationData


def least_squares(
    ansatz: StandardAnsatz,
    calibration_data: CalibrationData,
    initial_parameters: Parameters,
    num_iterations: int,
    device: Device,
) -> LeastSquaresOutput:
    freeze_model(ansatz)
    model = ansatz
    initial_parameters.requires_grad_(True).to(device)
    data = preprocess_calibration_data(calibration_data)
    inputs = data.inputs
    outputs = data.outputs
    num_data_points = data.num_data_points
    loss_metric = torch.nn.MSELoss()

    optimizer = torch.optim.LBFGS(
        params=initial_parameters,
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def calculate_model_outputs(parameters: Parameters) -> Tensor:
        model_inputs = torch.concat(
            (
                inputs.detach(),
                parameters.repeat((num_data_points, 1)),
            ),
            dim=1,
        ).to(device)
        return model(model_inputs)

    def loss_func(parameters: Parameters) -> Tensor:
        model_outputs = calculate_model_outputs(parameters)
        return loss_metric(model_outputs, outputs)

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_func(parameters)
        loss.backward()
        return loss.item()

    loss_hist = []
    parameters = initial_parameters
    for _ in range(num_iterations):
        loss = loss_func(parameters)
        optimizer.step(loss_func_closure)
        loss_hist.append(float(loss.detach().cpu().item()))

    identified_parameters = parameters.detach().cpu().numpy()
    print(f"Least squares: {identified_parameters}")

    return identified_parameters, loss_hist
