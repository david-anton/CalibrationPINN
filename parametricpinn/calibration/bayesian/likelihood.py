from typing import Protocol

import torch

from parametricpinn.calibration.data import PreprocessedCalibrationData
from parametricpinn.types import Device, Module, Tensor, TorchMultiNormalDist


class Likelihood(Protocol):
    def prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        pass


class CalibrationLikelihood:
    def __init__(
        self,
        model: Module,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._model = model
        self._data = data
        self._num_flattened_outputs = self._data.num_data_points * data.dim_outputs
        self._device = device
        self._likelihood = self._initialize_likelihood()

    def prob(self, parameters: Tensor) -> Tensor:
        with torch.no_grad():
            return self._prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        with torch.no_grad():
            return self._log_prob(parameters)

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        return torch.autograd.grad(
            self._log_prob(parameters),
            parameters,
            retain_graph=False,
            create_graph=False,
        )[0]

    def _prob(self, parameters: Tensor) -> Tensor:
        return torch.exp(self._log_prob(parameters))

    def _log_prob(self, parameters: Tensor) -> Tensor:
        residual = self._calculate_residual(parameters)
        return self._likelihood.log_prob(residual)

    def _initialize_likelihood(self) -> TorchMultiNormalDist:
        covariance_matrix = self._assemble_residual_covariance_matrix()
        return torch.distributions.MultivariateNormal(
            loc=torch.zeros(
                (self._num_flattened_outputs), dtype=torch.float64, device=self._device
            ),
            covariance_matrix=covariance_matrix,
        )

    def _assemble_residual_covariance_matrix(self) -> Tensor:
        return torch.diag(
            torch.full(
                (self._num_flattened_outputs,),
                self._data.std_noise**2,
                dtype=torch.float64,
                device=self._device,
            )
        )

    def _calculate_flattened_model_outputs(self, parameters: Tensor) -> Tensor:
        model_inputs = torch.concat(
            (
                self._data.inputs.detach(),
                parameters.repeat((self._data.num_data_points, 1)),
            ),
            dim=1,
        ).to(self._device)
        model_output = self._model(model_inputs)
        return model_output.ravel()

    def _calculate_residual(self, parameters: Tensor) -> Tensor:
        flattened_model_outputs = self._calculate_flattened_model_outputs(parameters)
        return flattened_model_outputs - self._data.outputs.ravel()
