import torch

from parametricpinn.ansatz import StandardAnsatz
from parametricpinn.bayesian.distributions import (
    IndependentMultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
)
from parametricpinn.calibration.base import (
    CalibrationData,
    PreprocessedCalibrationData,
    preprocess_calibration_data,
)
from parametricpinn.types import Device, Tensor


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
    preprocessed_data = preprocess_calibration_data(data)
    return PPINNLikelihood(
        ansatz=ansatz,
        data=preprocessed_data,
        device=device,
    )
