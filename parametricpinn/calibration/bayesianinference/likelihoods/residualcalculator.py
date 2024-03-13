import torch

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.types import Device, Tensor


class StandardResidualCalculator:
    def __init__(
        self,
        model: StandardAnsatz,
        device: Device,
    ):
        self._model = model.to(device)
        freeze_model(self._model)
        self._device = device

    def calculate_residuals(
        self, parameters: Tensor, inputs: Tensor, outputs: Tensor
    ) -> Tensor:
        inputs = inputs.detach().to(self._device)
        flattened_data_outputs = outputs.detach().to(self._device).ravel()
        flattened_model_outputs = self._calculate_flattened_model_outputs(
            parameters, inputs
        )
        return flattened_model_outputs - flattened_data_outputs

    def _calculate_flattened_model_outputs(
        self, parameters: Tensor, inputs: Tensor
    ) -> Tensor:
        num_data_points = len(inputs)
        model_inputs = torch.concat(
            (
                inputs,
                parameters.repeat((num_data_points, 1)),
            ),
            dim=1,
        )
        return self._model(model_inputs).ravel()
