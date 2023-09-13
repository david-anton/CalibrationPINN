from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn as nn

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.calibration.base import (
    CalibrationData,
    Parameters,
    PreprocessedCalibrationData,
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


class Predictor(nn.Module):
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
            initial_parameters.type(torch.float64), requires_grad=True
        )
        self._fixed_inputs = calibration_data.inputs.detach()
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
                self._parameter_inputs.expand(self._num_data_points, 1),
            ),
            dim=1,
        ).to(self._device)
        return self._model(model_inputs)


def least_squares(
    ansatz: StandardAnsatz,
    calibration_data: CalibrationData,
    initial_parameters: Parameters,
    num_iterations: int,
    device: Device,
) -> LeastSquaresOutput:
    preprocessed_data = preprocess_calibration_data(calibration_data)
    predictor = Predictor(ansatz, initial_parameters, preprocessed_data, device)
    outputs = preprocessed_data.outputs.detach()
    loss_metric = torch.nn.MSELoss()
    loss_weight = 1e6

    optimizer = torch.optim.LBFGS(
        params=predictor.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def loss_func() -> Tensor:
        model_outputs = predictor()
        return loss_weight * loss_metric(model_outputs, outputs)

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_func()
        loss.backward()
        return loss.item()

    loss_hist = []
    for _ in range(num_iterations):
        loss = loss_func()
        print(f"{_}: {loss}")
        optimizer.step(loss_func_closure)
        loss_hist.append(float(loss.detach().cpu().item()))

    identified_parameters = predictor.get_parameters_as_tensor().detach().cpu().numpy()

    return identified_parameters, loss_hist
