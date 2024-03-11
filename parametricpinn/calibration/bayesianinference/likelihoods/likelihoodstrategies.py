from typing import Optional, Protocol, TypeAlias

import torch

from parametricpinn.calibration.bayesianinference.likelihoods.residualcalculator import (
    StandardResidualCalculator,
)
from parametricpinn.calibration.data import PreprocessedCalibrationData
from parametricpinn.gps import GaussianProcess
from parametricpinn.statistics.distributions import (
    IndependentMultivariateNormalDistributon,
    MultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
    create_multivariate_normal_distribution,
)
from parametricpinn.types import Device, Tensor

Hyperparameters: TypeAlias = Tensor


class LikelihoodStrategy(Protocol):
    def log_prob(self, parameters: Tensor) -> Tensor:
        pass


class NoiseLikelihoodStrategy:
    def __init__(
        self,
        residual_calculator: StandardResidualCalculator,
        data: PreprocessedCalibrationData,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        self._residual_calculator = residual_calculator
        self._standard_deviation_noise = data.std_noise
        self._num_flattened_outputs = data.num_data_points * data.dim_outputs
        self._num_model_parameters = num_model_parameters
        self._device = device
        self._distribution = self._initialize_likelihood_distribution()

    def log_prob(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters[: self._num_model_parameters]
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        return self._distribution.log_prob(residuals)

    def flattened_log_probs(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters[: self._num_model_parameters]
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        return self._distribution.log_probs_individual(residuals)

    def _initialize_likelihood_distribution(
        self,
    ) -> IndependentMultivariateNormalDistributon:
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
            self._standard_deviation_noise,
            dtype=torch.float64,
            device=self._device,
        )


class NoiseAndModelErrorLikelihoodStrategy:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
        hyperparameters: Optional[Hyperparameters] = None,
    ) -> None:
        self._data = data
        self._data.inputs.detach().to(device)
        self._standard_deviation_noise = torch.tensor(data.std_noise, device=device)
        self._dim_outputs = data.dim_outputs
        self._num_data_points = data.num_data_points
        self._num_flattened_outputs = self._num_data_points * self._dim_outputs
        self._residual_calculator = residual_calculator
        self._num_model_parameters = num_model_parameters
        self._device = device
        self._hyperparameters = hyperparameters

    def log_prob(self, parameters: Tensor) -> Tensor:
        if self._hyperparameters is not None:
            model_parameters = parameters
            model_error_standard_deviations = self._hyperparameters
        else:
            model_parameters = parameters[: self._num_model_parameters]
            model_error_standard_deviations = parameters[-self._dim_outputs :]
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        distribution = self._initialize_likelihood_distribution(
            model_error_standard_deviations
        )
        return distribution.log_prob(residuals)

    def flattened_log_probs(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters[: self._num_model_parameters]
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        model_error_standard_deviation = parameters[-self._dim_outputs :]
        distribution = self._initialize_likelihood_distribution(
            model_error_standard_deviation
        )
        return distribution.log_probs_individual(residuals)

    def get_hyperparameters(self) -> Hyperparameters:
        return self._hyperparameters

    def set_hyperparameters(self, hyperparameters: Hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    def _initialize_likelihood_distribution(
        self, model_error_standard_deviations: Tensor
    ) -> IndependentMultivariateNormalDistributon:
        means = self._assemble_residual_means()
        standard_deviations = self._assemble_residual_standard_deviations(
            model_error_standard_deviations
        )
        return create_independent_multivariate_normal_distribution(
            means=means,
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def _assemble_residual_means(self) -> Tensor:
        return torch.zeros(
            (self._num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_residual_standard_deviations(
        self, model_error_standard_deviations: Tensor
    ) -> Tensor:
        noise_standard_deviations = self._assemble_noise_standard_deviations()
        error_standard_deviations = self._assemble_error_standard_deviations(
            model_error_standard_deviations
        )
        return torch.sqrt(
            noise_standard_deviations**2 + error_standard_deviations**2
        )

    def _assemble_noise_standard_deviations(self) -> Tensor:
        return self._standard_deviation_noise * torch.ones(
            (self._num_flattened_outputs,),
            dtype=torch.float64,
            device=self._device,
        )

    def _assemble_error_standard_deviations(
        self, error_standard_deviations: Tensor
    ) -> Tensor:
        normalized_standard_deviation = torch.ones(
            (self._num_data_points,),
            dtype=torch.float64,
            device=self._device,
        )
        return torch.concat(
            tuple(
                error_stddev * normalized_standard_deviation
                for error_stddev in error_standard_deviations
            )
        )


class NoiseAndModelErrorGPsLikelihoodStrategy:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        model_error_gp: GaussianProcess,
        device: Device,
        hyperparameters: Optional[Hyperparameters] = None,
    ) -> None:
        self._data = data
        self._data.inputs.detach().to(device)
        self._standard_deviation_noise = data.std_noise
        self._num_flattened_outputs = data.num_data_points * data.dim_outputs
        self._residual_calculator = residual_calculator
        self._num_model_parameters = num_model_parameters
        self._model_error_gp = model_error_gp.to(device)
        self._device = device
        self._hyperparameters = hyperparameters

    def log_prob(self, parameters: Tensor) -> Tensor:
        if self._hyperparameters is not None:
            model_parameters = parameters
            gp_parameters = self._hyperparameters
        else:
            model_parameters = parameters[: self._num_model_parameters]
            gp_parameters = parameters[self._num_model_parameters :]
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        distribution = self._initialize_likelihood_distribution(gp_parameters)
        return distribution.log_prob(residuals)

    def get_hyperparameters(self) -> Hyperparameters:
        return self._hyperparameters

    def set_hyperparameters(self, hyperparameters: Hyperparameters) -> None:
        self._hyperparameters = hyperparameters

    def _initialize_likelihood_distribution(
        self, gp_parameters: Tensor
    ) -> MultivariateNormalDistributon:
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
        self._model_error_gp.set_parameters(gp_parameters)
        inputs = self._data.inputs
        return self._model_error_gp.forward_kernel(inputs, inputs)
