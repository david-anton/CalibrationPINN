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
from parametricpinn.types import Device, Parameter, Tensor


class LikelihoodStrategy(Protocol):
    def log_prob(self, parameters: Tensor) -> Tensor:
        pass


### Noise
class NoiseLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        residual_calculator: StandardResidualCalculator,
        data: PreprocessedCalibrationData,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._residual_calculator = residual_calculator
        self._standard_deviation_noise = data.std_noise
        self._num_flattened_outputs = data.num_data_points * data.dim_outputs
        self._num_model_parameters = num_model_parameters
        self._device = device
        self._distribution = self._initialize_likelihood_distribution()

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

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


### Noise and model error
class NoiseAndModelErrorLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._noise_standard_deviation = torch.tensor(data.std_noise, device=device)
        self._num_data_points = data.num_data_points
        self._num_flattened_outputs = self._num_data_points * data.dim_outputs
        self._device = device

    def initialize(
        self, model_error_standard_deviation_parameters: Tensor | Parameter
    ) -> IndependentMultivariateNormalDistributon:
        means = self._assemble_means()
        standard_deviations = self._assemble_standard_deviations(
            model_error_standard_deviation_parameters
        )
        return create_independent_multivariate_normal_distribution(
            means=means,
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def _assemble_means(self) -> Tensor:
        return torch.zeros(
            (self._num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_standard_deviations(
        self, model_error_standard_deviation_parameters: Tensor | Parameter
    ) -> Tensor:
        noise_standard_deviations = self._assemble_noise_standard_deviations()
        model_error_standard_deviations = self._assemble_error_standard_deviations(
            model_error_standard_deviation_parameters
        )
        return torch.sqrt(
            noise_standard_deviations**2 + model_error_standard_deviations**2
        )

    def _assemble_noise_standard_deviations(self) -> Tensor:
        return self._noise_standard_deviation * torch.ones(
            (self._num_flattened_outputs,),
            dtype=torch.float64,
            device=self._device,
        )

    def _assemble_error_standard_deviations(
        self, model_error_standard_deviation_parameters: Tensor | Parameter
    ) -> Tensor:
        normalized_standard_deviation = torch.ones(
            (self._num_data_points,),
            dtype=torch.float64,
            device=self._device,
        )
        return torch.concat(
            tuple(
                error_stdandard_deviation_parameter * normalized_standard_deviation
                for error_stdandard_deviation_parameter in model_error_standard_deviation_parameters
            )
        )


class NoiseAndModelErrorSamplingLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._distribution = NoiseAndModelErrorLikelihoodDistribution(data, device)
        self._data = data
        self._data.inputs.detach().to(device)
        self._residual_calculator = residual_calculator
        self._num_model_parameters = num_model_parameters

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        (
            model_parameters,
            model_error_standard_deviation_parameters,
        ) = self._split_parameters(parameters)
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        distribution = self._distribution.initialize(
            model_error_standard_deviation_parameters
        )
        return distribution.log_prob(residuals)

    def flattened_log_probs(self, parameters: Tensor) -> Tensor:
        (
            model_parameters,
            model_error_standard_deviation_parameters,
        ) = self._split_parameters(parameters)
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        distribution = self._distribution.initialize(
            model_error_standard_deviation_parameters
        )
        return distribution.log_probs_individual(residuals)

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        model_parameters = parameters[: self._num_model_parameters]
        model_error_standard_deviation_parameters = parameters[
            self._num_model_parameters :
        ]
        return model_parameters, model_error_standard_deviation_parameters


class NoiseAndModelErrorOptimizeLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        initial_model_error_standard_deviations: Tensor,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._model_error_standard_deviation_parameters = torch.nn.Parameter(
            initial_model_error_standard_deviations, requires_grad=True
        )
        self._distribution = NoiseAndModelErrorLikelihoodDistribution(data, device)
        self._data = data
        self._data.inputs.detach().to(device)
        self._residual_calculator = residual_calculator
        self._num_model_parameters = num_model_parameters

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        distribution = self._distribution.initialize(
            self._model_error_standard_deviation_parameters
        )
        return distribution.log_prob(residuals)

    def flattened_log_probs(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        distribution = self._distribution.initialize(
            self._model_error_standard_deviation_parameters
        )
        return distribution.log_probs_individual(residuals)


### Noise and model error GPs
class NoiseAndModelErrorGPsLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._data = data
        self._data.inputs.detach().to(device)
        self._standard_deviation_noise = torch.tensor(data.std_noise, device=device)
        self._num_data_points = data.num_data_points
        self._num_flattened_outputs = self._num_data_points * data.dim_outputs
        self._device = device

    def _initialize(
        self, model_error_gp: GaussianProcess
    ) -> MultivariateNormalDistributon:
        means = self._assemble_means()
        covariance_matrix = self._assemble_covariance_matrix(model_error_gp)
        return create_multivariate_normal_distribution(
            means=means, covariance_matrix=covariance_matrix, device=self._device
        )

    def _assemble_means(self) -> Tensor:
        return torch.zeros(
            (self._num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_covariance_matrix(self, model_error_gp: GaussianProcess) -> Tensor:
        noise_covariance = self._assemble_noise_covariance_matrix()
        error_covariance = self._assemble_error_covariance_matrix(model_error_gp)
        return noise_covariance + error_covariance

    def _assemble_noise_covariance_matrix(self) -> Tensor:
        return self._standard_deviation_noise**2 * torch.eye(
            self._num_flattened_outputs, dtype=torch.float64, device=self._device
        )

    def _assemble_error_covariance_matrix(
        self, model_error_gp: GaussianProcess
    ) -> Tensor:
        inputs = self._data.inputs
        return model_error_gp.forward_kernel(inputs, inputs)


class NoiseAndModelErrorGPsSamplingLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        model_error_gp: GaussianProcess,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._model_error_gp = model_error_gp.to(device)
        self._distribution = NoiseAndModelErrorGPsLikelihoodDistribution(data, device)
        self._data = data
        self._data.inputs.detach().to(device)
        self._residual_calculator = residual_calculator
        self._num_model_parameters = num_model_parameters
        self._device = device

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        model_parameters, model_error_gp_parameters = self._split_parameters(parameters)
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        self._model_error_gp.set_parameters(model_error_gp_parameters)
        distribution = self._distribution._initialize(self._model_error_gp)
        return distribution.log_prob(residuals)

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        model_parameters = parameters[: self._num_model_parameters]
        model_error_gp_parameters = parameters[self._num_model_parameters :]
        return model_parameters, model_error_gp_parameters

        self._model_error_gp.set_parameters(hyperparameters)


class NoiseAndModelErrorGPsOptimizeLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        model_error_gp: GaussianProcess,
        initial_model_error_gp_parameters: Tensor,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._model_error_gp = model_error_gp.to(device)
        self._model_error_gp.set_parameters(initial_model_error_gp_parameters)
        self._distribution = NoiseAndModelErrorGPsLikelihoodDistribution(data, device)
        self._data = data
        self._data.inputs.detach().to(device)
        self._residual_calculator = residual_calculator
        self._num_model_parameters = num_model_parameters
        self._device = device

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        distribution = self._distribution._initialize(self._model_error_gp)
        return distribution.log_prob(residuals)
