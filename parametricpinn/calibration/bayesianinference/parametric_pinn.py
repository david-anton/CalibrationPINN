from typing import Protocol, TypeAlias, Union

import torch

from parametricpinn.ansatz import BayesianAnsatz, StandardAnsatz
from parametricpinn.calibration.base import (
    CalibrationData,
    PreprocessedCalibrationData,
    preprocess_calibration_data,
)
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.gps import ZeroMeanScaledRBFKernelGP
from parametricpinn.statistics.distributions import (
    IndependentMultivariateNormalDistributon,
    MultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
    create_multivariate_normal_distribution,
)
from parametricpinn.types import Device, Tensor

LikelihoodDistributions: TypeAlias = Union[
    IndependentMultivariateNormalDistributon, MultivariateNormalDistributon
]


class LikelihoodStrategy(Protocol):
    def log_prob(
        self, residuals: Tensor, parameters: Tensor, num_model_parameters: int
    ) -> Tensor:
        pass


class NoiseLikelihoodStrategy:
    def __init__(self, data: PreprocessedCalibrationData, device: Device) -> None:
        self.standard_deviation_noise = data.std_noise
        self._num_flattened_outputs = data.num_data_points * data.dim_outputs
        self._device = device

    def log_prob(
        self, residuals: Tensor, parameters: Tensor, num_model_parameters: int
    ) -> Tensor:
        distribution = self._initialize_likelihood_distribution()
        return distribution.log_prob(residuals)

    def _initialize_likelihood_distribution(self) -> LikelihoodDistributions:
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
            self.standard_deviation_noise,
            dtype=torch.float64,
            device=self._device,
        )


class NoiseAndModelErrorLikelihoodStrategy:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        model_error_gp: ZeroMeanScaledRBFKernelGP,
        device: Device,
    ) -> None:
        self._data = data
        self._data.inputs.detach().to(device)
        self._standard_deviation_noise = data.std_noise
        self._num_flattened_outputs = data.num_data_points * data.dim_outputs
        self._model_error_gp = model_error_gp.to(device)
        self._device = device

    def log_prob(
        self, residuals: Tensor, parameters: Tensor, num_model_parameters: int
    ) -> Tensor:
        gp_parameters = parameters[num_model_parameters:]
        distribution = self._initialize_likelihood_distribution(gp_parameters)
        return distribution.log_prob(residuals)

    def _initialize_likelihood_distribution(
        self, gp_parameters: Tensor
    ) -> LikelihoodDistributions:
        means = self._assemble_residual_means()
        covariance_matrix = self._calculate_residual_covariance_matrix(gp_parameters)
        return create_multivariate_normal_distribution(
            means=means, covariance_matrix=covariance_matrix, device=self._device
        )

    def _assemble_residual_means(self) -> Tensor:
        return torch.zeros(
            (self._num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _calculate_residual_covariance_matrix(self, gp_parameters: Tensor) -> Tensor:
        noise_covar = self._assemble_noise_covar_matrix()
        model_error_covar = self._calculate_model_error_covar_matrix(gp_parameters)
        return noise_covar + model_error_covar

    def _assemble_noise_covar_matrix(self) -> Tensor:
        return self._standard_deviation_noise**2 * torch.eye(
            self._num_flattened_outputs, dtype=torch.float64, device=self._device
        )

    def _calculate_model_error_covar_matrix(self, gp_parameters: Tensor) -> Tensor:
        # It is assumed that the covariance structure of the error is identical in all dimensions
        # and that the errors in the individual dimensions are not dependent on each other
        # (covariance between dimensions = 0).
        output_scale = gp_parameters[0]
        length_scale = gp_parameters[1]
        self._model_error_gp.set_covariance_parameters(
            torch.tensor([output_scale, length_scale])
        )
        inputs = self._data.inputs
        covar_matrix = self._model_error_gp.forward_kernel(inputs, inputs)
        return self._apply_model_error_covar_matrix_for_all_output_dimensions(
            covar_matrix
        )

    def _apply_model_error_covar_matrix_for_all_output_dimensions(
        self, covar_matrix: Tensor
    ) -> Tensor:
        flattened_covar_matrix = torch.zeros(
            (self._num_flattened_outputs, self._num_flattened_outputs),
            dtype=torch.float64,
            device=self._device,
        )
        num_output_dims = self._data.dim_outputs
        num_data_points = self._data.num_data_points
        start_index = 0
        for _ in range(num_output_dims - 1):
            index_slice = slice(start_index, start_index + num_data_points)
            flattened_covar_matrix[index_slice, index_slice] = covar_matrix
            start_index += num_data_points
        flattened_covar_matrix[start_index:, start_index:] = covar_matrix
        return flattened_covar_matrix


class StandardPPINNLikelihood:
    def __init__(
        self,
        likelihood_strategy: LikelihoodStrategy,
        model: StandardAnsatz,
        num_model_parameters: int,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._likelihood_strategy = likelihood_strategy
        self._model = model.to(device)
        freeze_model(self._model)
        self._num_model_parameters = num_model_parameters
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
        model_parameters = parameters[0 : self._num_model_parameters]
        residuals = self._calculate_residuals(model_parameters)
        return self._likelihood_strategy.log_prob(
            residuals, parameters, self._num_model_parameters
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


def create_standard_ppinn_likelihood_for_noise(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    likelihood_strategy = NoiseLikelihoodStrategy(data=preprocessed_data, device=device)
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        model=model,
        num_model_parameters=num_model_parameters,
        data=preprocessed_data,
        device=device,
    )


def create_standard_ppinn_likelihood_for_noise_and_model_error(
    model: StandardAnsatz,
    num_model_parameters: int,
    model_error_gp: ZeroMeanScaledRBFKernelGP,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    likelihood_strategy = NoiseAndModelErrorLikelihoodStrategy(
        data=preprocessed_data, model_error_gp=model_error_gp, device=device
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        model=model,
        num_model_parameters=num_model_parameters,
        data=preprocessed_data,
        device=device,
    )


class BayesianPPINNLikelihood:
    def __init__(
        self,
        model: BayesianAnsatz,
        model_parameter_samples: Tensor,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._model = model.to(device)
        self._model_parameter_samples = model_parameter_samples.to(device)
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
        means, stddevs = self._calculate_model_output_means_and_stddevs(parameters)
        flattened_model_means = means.ravel()
        flattened_model_stddevs = stddevs.ravel()
        residuals = self._calculate_residuals(flattened_model_means)
        likelihood = self._initialize_likelihood(flattened_model_stddevs)
        return likelihood.log_prob(residuals)

    def _calculate_model_output_means_and_stddevs(
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
        return torch.sqrt(flattened_noise_stddevs**2 + flattened_model_stddevs**2)

    def _calculate_residuals(self, flattened_means: Tensor) -> Tensor:
        flattened_outputs = self._data.outputs.ravel()
        return flattened_means - flattened_outputs


def create_bayesian_ppinn_likelihood_for_noise(
    model: BayesianAnsatz,
    model_parameter_samples: Tensor,
    data: CalibrationData,
    device: Device,
) -> BayesianPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    return BayesianPPINNLikelihood(
        model=model,
        model_parameter_samples=model_parameter_samples,
        data=preprocessed_data,
        device=device,
    )
