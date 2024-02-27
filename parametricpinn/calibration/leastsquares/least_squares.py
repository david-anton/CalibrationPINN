from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn as nn

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.calibration.config import CalibrationConfig
from parametricpinn.calibration.data import (
    CalibrationData,
    Parameters,
    PreprocessedCalibrationData,
    preprocess_calibration_data,
)
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
    resdiual_weights: Tensor


class ModelClosure(nn.Module):
    def __init__(
        self,
        ansatz: StandardAnsatz,
        initial_parameters: Parameters,
        calibration_data: PreprocessedCalibrationData,
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
        return self._parameter_inputs.data

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
    calibration_data: CalibrationData,
    initial_parameters: Parameters,
    num_iterations: int,
    residual_weights: Tensor,
    device: Device,
) -> LeastSquaresOutput:
    initial_parameters = initial_parameters.clone()
    preprocessed_data = preprocess_calibration_data(calibration_data)
    model_closure = ModelClosure(ansatz, initial_parameters, preprocessed_data, device)
    flattened_outputs = preprocessed_data.outputs.ravel().detach().to(device)
    residual_weights = residual_weights.to(device)

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

    def loss_func() -> Tensor:
        flattened_model_outputs = model_closure().ravel()
        residuals = flattened_model_outputs - flattened_outputs
        weighted_residuals = residual_weights * residuals
        return (1 / 2) * torch.matmul(
            weighted_residuals,
            torch.transpose(torch.unsqueeze(weighted_residuals, dim=0), 0, 1),
        )

    def loss_func_closure() -> float:
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

    return identified_parameters, loss_hist
