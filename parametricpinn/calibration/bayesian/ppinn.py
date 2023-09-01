from dataclasses import dataclass

import torch

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.calibration.bayesian.distributions import (
    IndependentMultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
)
from parametricpinn.errors import UnvalidCalibrationDataError
from parametricpinn.types import Device, Tensor


@dataclass
class CalibrationData:
    inputs: Tensor
    outputs: Tensor
    std_noise: float


@dataclass
class PreprocessedCalibrationData(CalibrationData):
    num_data_points: int
    dim_outputs: int


class PPINNLikelihood:
    def __init__(
        self,
        ansatz: StandardAnsatz,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._model = ansatz
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
        residuals = self._calculate_residuals(parameters)
        return self._likelihood.log_prob(residuals)

    def _initialize_likelihood(self) -> IndependentMultivariateNormalDistributon:
        means = self._assemble_residual_means()
        standard_deviations = self._assemble_residual_standard_deviations()
        return create_independent_multivariate_normal_distribution(
            means=means,
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def _assemble_residual_means(self) -> Tensor:
        return torch.zeros(
            (self._num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_residual_standard_deviations(self) -> Tensor:
        return torch.full(
            (self._num_flattened_outputs,),
            self._data.std_noise,
            dtype=torch.float64,
            device=self._device,
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

    def _calculate_residuals(self, parameters: Tensor) -> Tensor:
        flattened_model_outputs = self._calculate_flattened_model_outputs(parameters)
        return flattened_model_outputs - self._data.outputs.ravel()


def create_ppinn_likelihood(
    ansatz: StandardAnsatz, data: CalibrationData, device: Device
) -> PPINNLikelihood:
    preprocessed_data = _preprocess_calibration_data(data)
    return PPINNLikelihood(
        ansatz=ansatz,
        data=preprocessed_data,
        device=device,
    )


def _preprocess_calibration_data(data: CalibrationData) -> PreprocessedCalibrationData:
    _validate_calibration_data(data)
    outputs = data.outputs
    num_data_points = outputs.size()[0]
    dim_outputs = outputs.size()[1]

    return PreprocessedCalibrationData(
        inputs=data.inputs,
        outputs=data.outputs,
        std_noise=data.std_noise,
        num_data_points=num_data_points,
        dim_outputs=dim_outputs,
    )


def _validate_calibration_data(calibration_data: CalibrationData) -> None:
    inputs = calibration_data.inputs
    outputs = calibration_data.outputs
    if not inputs.size()[0] == outputs.size()[0]:
        raise UnvalidCalibrationDataError(
            "Size of input and output data does not match."
        )
