from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn as nn

from calibrationpinn.ansatz import StandardAnsatz
from calibrationpinn.calibration.config import CalibrationConfig
from calibrationpinn.calibration.data import (
    CalibrationData,
    ConcatenatedCalibrationData,
    Parameters,
    concatenate_calibration_data,
)
from calibrationpinn.calibration.utility import freeze_model
from calibrationpinn.types import Device, NPArray, Tensor

LeastSquaresOutput: TypeAlias = tuple[NPArray, list[float], int]
LeastSquaresFunc: TypeAlias = Callable[
    [
        StandardAnsatz,
        CalibrationData | ConcatenatedCalibrationData,
        Parameters,
        int,
        Device,
    ],
    LeastSquaresOutput,
]


@dataclass
class LeastSquaresConfig(CalibrationConfig):
    ansatz: StandardAnsatz
    calibration_data: CalibrationData | ConcatenatedCalibrationData
    resdiual_weights: Tensor


class ModelClosure(nn.Module):
    def __init__(
        self,
        ansatz: StandardAnsatz,
        initial_parameters: Parameters,
        calibration_data: ConcatenatedCalibrationData,
        device: Device,
    ) -> None:
        super().__init__()
        self._model = ansatz
        freeze_model(self._model)
        self._parameter_inputs = nn.Parameter(
            initial_parameters.type(torch.float64).to(device), requires_grad=True
        )
        self._fixed_inputs = calibration_data.inputs.detach().to(device)
        self._num_data_points = calibration_data.num_data_points
        self._device = device

    def forward(self) -> Tensor:
        return self._calculate_model_outputs()

    def get_parameters_as_tensor(self) -> Parameters:
        return self._parameter_inputs.data.detach()

    def _calculate_model_outputs(self) -> Tensor:
        model_inputs = torch.concat(
            (
                self._fixed_inputs,
                self._parameter_inputs.repeat((self._num_data_points, 1)),
            ),
            dim=1,
        ).to(self._device)
        return self._model(model_inputs)


def least_squares(
    ansatz: StandardAnsatz,
    calibration_data: CalibrationData | ConcatenatedCalibrationData,
    initial_parameters: Parameters,
    num_iterations: int,
    residual_weights: Tensor,
    device: Device,
) -> LeastSquaresOutput:
    if isinstance(calibration_data, CalibrationData):
        calibration_data = concatenate_calibration_data(calibration_data)
    initial_parameters = initial_parameters.clone()
    model_closure = ModelClosure(ansatz, initial_parameters, calibration_data, device)
    flattened_outputs = flatten(calibration_data.outputs.detach().to(device))
    flattened_residual_weights = flatten(residual_weights.to(device))

    optimizer = torch.optim.LBFGS(
        params=model_closure.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    class FunctionCallCounter:
        def __init__(self):
            self.num_function_calls = 0

        def increment(self):
            self.num_function_calls += 1

    loss_function_call_counter = FunctionCallCounter()

    def loss_func() -> Tensor:
        flattened_model_outputs = flatten(model_closure())
        residuals = flattened_model_outputs - flattened_outputs
        weighted_residuals = flattened_residual_weights * residuals
        return (1 / 2) * torch.matmul(
            weighted_residuals,
            torch.transpose(torch.unsqueeze(weighted_residuals, dim=0), 0, 1),
        )

    def loss_func_closure() -> float:
        loss_function_call_counter.increment()
        optimizer.zero_grad()
        loss = loss_func()
        loss.backward()
        return loss.item()

    loss_hist = []
    for _ in range(num_iterations):
        loss = loss_func()
        optimizer.step(loss_func_closure)
        loss_hist.append(float(loss.detach().cpu().item()))

    identified_parameters = (
        model_closure.get_parameters_as_tensor().detach().cpu().numpy()
    )
    num_loss_function_calls = loss_function_call_counter.num_function_calls

    return identified_parameters, loss_hist, num_loss_function_calls


def flatten(x: Tensor) -> Tensor:
    return torch.transpose(x, 1, 0).ravel()
