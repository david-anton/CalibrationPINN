import torch

from parametricpinn.ansatz import BayesianAnsatz, StandardAnsatz
from parametricpinn.bayesian.distributions import (
    IndependentMultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
)
from parametricpinn.calibration.base import (
    CalibrationData,
    PreprocessedCalibrationData,
    preprocess_calibration_data,
)
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.types import Device, Tensor


class StandardPPINNLikelihood:
    def __init__(
        self,
        ansatz: StandardAnsatz,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._model = ansatz.to(device)
        freeze_model(self._model)
        self._data = data
        self._data.inputs.detach().to(device)
        self._data.outputs.detach().to(device)
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
        parameters.to(self._device)
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

    def _calculate_residuals(self, parameters: Tensor) -> Tensor:
        flattened_model_outputs = self._calculate_flattened_model_outputs(parameters)
        return flattened_model_outputs - self._data.outputs.ravel()

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


def create_standard_ppinn_likelihood(
    ansatz: StandardAnsatz, data: CalibrationData, device: Device
) -> StandardPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    return StandardPPINNLikelihood(
        ansatz=ansatz,
        data=preprocessed_data,
        device=device,
    )


class BayesianPPINNLikelihood:
    def __init__(
        self,
        ansatz: BayesianAnsatz,
        ansatz_parameter_samples: Tensor,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._model = ansatz.to(device)
        self._model_parameter_samples = ansatz_parameter_samples.to(device)
        self._data = data
        self._data.inputs.detach().to(device)
        self._data.outputs.detach().to(device)
        self._num_flattened_outputs = self._data.num_data_points * data.dim_outputs
        self._device = device

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
        parameters.to(self._device)
        means, stddevs = self._calculate_means_and_stddevs_of_model_output(parameters)
        flattened_model_means = means.ravel()
        flattened_model_stddevs = stddevs.ravel()
        residuals = self._calculate_residuals(flattened_model_means)
        likelihood = self._initialize_likelihood(flattened_model_stddevs)
        return likelihood.log_prob(residuals)

    def _calculate_means_and_stddevs_of_model_output(
        self, parameters: Tensor
    ) -> tuple[Tensor, Tensor]:
        model_inputs = torch.concat(
            (
                self._data.inputs,
                parameters.repeat((self._data.num_data_points, 1)),
            ),
            dim=1,
        )
        means, stddevs = self._model.predict_normal_distribution(
            model_inputs, self._model_parameter_samples
        )
        return means, stddevs

    def _initialize_likelihood(
        self, flattened_model_stddevs: Tensor
    ) -> IndependentMultivariateNormalDistributon:
        residual_means = self._assemble_residual_means()
        residual_stddevs = self._assemble_residual_standard_deviations(
            flattened_model_stddevs
        )
        return create_independent_multivariate_normal_distribution(
            means=residual_means,
            standard_deviations=residual_stddevs,
            device=self._device,
        )

    def _assemble_residual_means(self) -> Tensor:
        return torch.zeros(
            (self._num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_residual_standard_deviations(
        self, flattened_model_stddevs: Tensor
    ) -> Tensor:
        flattened_noise_stddevs = torch.full(
            (self._num_flattened_outputs,),
            self._data.std_noise,
            dtype=torch.float64,
            device=self._device,
        )
        return flattened_noise_stddevs + flattened_model_stddevs

    def _calculate_residuals(self, flattened_means: Tensor) -> Tensor:
        flattened_outputs = self._data.outputs.ravel()
        return flattened_means - flattened_outputs


def create_bayesian_ppinn_likelihood(
    ansatz: BayesianAnsatz,
    ansatz_parameter_samples: Tensor,
    data: CalibrationData,
    device: Device,
) -> BayesianPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    return BayesianPPINNLikelihood(
        ansatz=ansatz,
        ansatz_parameter_samples=ansatz_parameter_samples,
        data=preprocessed_data,
        device=device,
    )
