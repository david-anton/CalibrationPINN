import torch

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.calibration.data import PreprocessedCalibrationData
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.types import Device, Tensor


class StandardResidualCalculator:
    def __init__(
        self,
        model: StandardAnsatz,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._model = model.to(device)
        freeze_model(self._model)
        self._data = data
        self._data.inputs.detach().to(device)
        self._data.outputs.detach().to(device)
        self._flattened_data_outputs = self._data.outputs.ravel()
        self._num_flattened_data_outputs = len(self._flattened_data_outputs)
        self._device = device

    def calculate_residuals(self, parameters: Tensor) -> Tensor:
        flattened_model_outputs = self._calculate_flattened_model_outputs(parameters)
        return flattened_model_outputs - self._flattened_data_outputs

    def _calculate_flattened_model_outputs(self, parameters: Tensor) -> Tensor:
        model_inputs = torch.concat(
            (
                self._data.inputs,
                parameters.repeat((self._data.num_data_points, 1)),
            ),
            dim=1,
        )
        model_output = self._model(model_inputs)
        return model_output.ravel()
