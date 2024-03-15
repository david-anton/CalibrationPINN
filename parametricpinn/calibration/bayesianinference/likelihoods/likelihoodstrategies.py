from typing import Protocol

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
from parametricpinn.types import Device, Module, Parameter, Tensor


class LikelihoodStrategy(Protocol):
    def forward(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        pass


### Noise
class NoiseLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._noise_standard_deviation = torch.tensor(data.std_noise, device=device)
        self._device = device

    def initialize(
        self,
        num_flattened_outputs: int,
    ) -> IndependentMultivariateNormalDistributon:
        means = self._assemble_means(num_flattened_outputs)
        standard_deviations = self._assemble_standard_deviations(num_flattened_outputs)
        return create_independent_multivariate_normal_distribution(
            means=means,
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def _assemble_means(self, num_flattened_outputs: int) -> Tensor:
        return torch.zeros(
            (num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_standard_deviations(self, num_flattened_outputs: int) -> Tensor:
        return self._noise_standard_deviation * torch.ones(
            (num_flattened_outputs,),
            dtype=torch.float64,
            device=self._device,
        )


class NoiseLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        residual_calculator: StandardResidualCalculator,
        data: PreprocessedCalibrationData,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._distribution = NoiseLikelihoodDistribution(data, device)
        self._residual_calculator = residual_calculator
        self._inputs_sets = tuple(inputs.detach().to(device) for inputs in data.inputs)
        self._outputs_sets = tuple(
            outputs.detach().to(device) for outputs in data.outputs
        )
        self._num_data_sets = data.num_data_sets
        self._num_data_points_per_set = data.num_data_points_per_set
        self._dim_outputs = data.dim_outputs
        self._standard_deviation_noise = data.std_noise
        self._num_model_parameters = num_model_parameters
        self._device = device

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = self.log_probs_individual(parameters)
        return sum_individual_log_probs(log_probs, self._num_data_sets)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        return self._calculate_log_probs_for_data_sets(model_parameters)

    def _calculate_log_probs_for_data_sets(self, model_parameters: Tensor) -> Tensor:
        log_probs = tuple(
            self._calculate_one_log_prob_for_data_set(
                model_parameters,
                self._inputs_sets[data_set_index],
                self._outputs_sets[data_set_index],
                self._num_data_points_per_set[data_set_index],
            )
            for data_set_index in range(self._num_data_sets)
        )
        if self._num_data_sets == 1:
            return log_probs[0]
        else:
            return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        num_flattened_outputs = num_data_points * self._dim_outputs
        distribution = self._distribution.initialize(num_flattened_outputs)
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)


### Noise and model error
class NoiseAndModelErrorLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._noise_standard_deviation = torch.tensor(data.std_noise, device=device)
        self._device = device

    def initialize(
        self,
        model_error_standard_deviation_parameters: Tensor | Parameter,
        num_data_points: int,
        num_flattened_outputs: int,
    ) -> IndependentMultivariateNormalDistributon:
        means = self._assemble_means(num_flattened_outputs)
        standard_deviations = self._assemble_standard_deviations(
            model_error_standard_deviation_parameters,
            num_data_points,
            num_flattened_outputs,
        )
        return create_independent_multivariate_normal_distribution(
            means=means,
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def _assemble_means(self, num_flattened_outputs: int) -> Tensor:
        return torch.zeros(
            (num_flattened_outputs), dtype=torch.float64, device=self._device
        )

    def _assemble_standard_deviations(
        self,
        model_error_standard_deviation_parameters: Tensor | Parameter,
        num_data_points: int,
        num_flattened_outputs: int,
    ) -> Tensor:
        noise_standard_deviations = self._assemble_noise_standard_deviations(
            num_flattened_outputs
        )
        model_error_standard_deviations = self._assemble_error_standard_deviations(
            model_error_standard_deviation_parameters, num_data_points
        )
        return torch.sqrt(
            noise_standard_deviations**2 + model_error_standard_deviations**2
        )

    def _assemble_noise_standard_deviations(self, num_flattened_outputs: int) -> Tensor:
        return self._noise_standard_deviation * torch.ones(
            (num_flattened_outputs,),
            dtype=torch.float64,
            device=self._device,
        )

    def _assemble_error_standard_deviations(
        self,
        model_error_standard_deviation_parameters: Tensor | Parameter,
        num_data_points: int,
    ) -> Tensor:
        normalized_standard_deviation = torch.ones(
            (num_data_points,),
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
        self._residual_calculator = residual_calculator
        self._inputs_sets = tuple(inputs.detach().to(device) for inputs in data.inputs)
        self._outputs_sets = tuple(
            outputs.detach().to(device) for outputs in data.outputs
        )
        self._num_data_sets = data.num_data_sets
        self._num_data_points_per_set = data.num_data_points_per_set
        self._dim_outputs = data.dim_outputs
        self._standard_deviation_noise = data.std_noise
        self._num_model_parameters = num_model_parameters
        self._device = device

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = self.log_probs_individual(parameters)
        return sum_individual_log_probs(log_probs, self._num_data_sets)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        (
            model_parameters,
            model_error_standard_deviation_parameters,
        ) = self._split_parameters(parameters)
        return self._calculate_log_probs_for_data_sets(
            model_parameters, model_error_standard_deviation_parameters
        )

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        model_parameters = parameters[: self._num_model_parameters]
        model_error_standard_deviation_parameters = parameters[
            self._num_model_parameters :
        ]
        return model_parameters, model_error_standard_deviation_parameters

    def _calculate_log_probs_for_data_sets(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
    ) -> Tensor:
        log_probs = tuple(
            self._calculate_one_log_prob_for_data_set(
                model_parameters,
                model_error_standard_deviation_parameters,
                self._inputs_sets[data_set_index],
                self._outputs_sets[data_set_index],
                self._num_data_points_per_set[data_set_index],
            )
            for data_set_index in range(self._num_data_sets)
        )
        if self._num_data_sets == 1:
            return log_probs[0]
        else:
            return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        num_flattened_outputs = num_data_points * self._dim_outputs
        distribution = self._distribution.initialize(
            model_error_standard_deviation_parameters,
            num_data_points,
            num_flattened_outputs,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)


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
        self._residual_calculator = residual_calculator
        self._inputs_sets = tuple(inputs.detach().to(device) for inputs in data.inputs)
        self._outputs_sets = tuple(
            outputs.detach().to(device) for outputs in data.outputs
        )
        self._num_data_sets = data.num_data_sets
        self._num_data_points_per_set = data.num_data_points_per_set
        self._dim_outputs = data.dim_outputs
        self._standard_deviation_noise = data.std_noise
        self._num_model_parameters = num_model_parameters
        self._device = device

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = self.log_probs_individual(parameters)
        return sum_individual_log_probs(log_probs, self._num_data_sets)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        return self._calculate_log_probs_for_data_sets(
            model_parameters, self._model_error_standard_deviation_parameters
        )

    def _calculate_log_probs_for_data_sets(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
    ) -> Tensor:
        log_probs = tuple(
            self._calculate_one_log_prob_for_data_set(
                model_parameters,
                model_error_standard_deviation_parameters,
                self._inputs_sets[data_set_index],
                self._outputs_sets[data_set_index],
                self._num_data_points_per_set[data_set_index],
            )
            for data_set_index in range(self._num_data_sets)
        )
        if self._num_data_sets == 1:
            return log_probs[0]
        else:
            return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        num_flattened_outputs = num_data_points * self._dim_outputs
        distribution = self._distribution.initialize(
            model_error_standard_deviation_parameters,
            num_data_points,
            num_flattened_outputs,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)


### Noise and model error GPs
class NoiseAndModelErrorGPsLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._standard_deviation_noise = torch.tensor(data.std_noise, device=device)
        self._device = device

    def initialize(
        self,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        num_flattened_outputs: int,
    ) -> MultivariateNormalDistributon:
        means = self._assemble_means(num_flattened_outputs)
        covariance_matrix = self._assemble_covariance_matrix(
            model_error_gp, inputs, num_flattened_outputs
        )
        return create_multivariate_normal_distribution(
            means=means, covariance_matrix=covariance_matrix, device=self._device
        )

    def _assemble_means(self, num_flattened_outputs: int) -> Tensor:
        return torch.zeros(
            (num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_covariance_matrix(
        self,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        num_flattened_outputs: int,
    ) -> Tensor:
        noise_covariance = self._assemble_noise_covariance_matrix(num_flattened_outputs)
        error_covariance = self._assemble_error_covariance_matrix(
            model_error_gp, inputs
        )
        return noise_covariance + error_covariance

    def _assemble_noise_covariance_matrix(self, num_flattened_outputs: int) -> Tensor:
        return self._standard_deviation_noise**2 * torch.eye(
            num_flattened_outputs, dtype=torch.float64, device=self._device
        )

    def _assemble_error_covariance_matrix(
        self, model_error_gp: GaussianProcess, inputs: Tensor
    ) -> Tensor:
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
        self._residual_calculator = residual_calculator
        self._inputs_sets = tuple(inputs.detach().to(device) for inputs in data.inputs)
        self._outputs_sets = tuple(
            outputs.detach().to(device) for outputs in data.outputs
        )
        self._num_data_sets = data.num_data_sets
        self._num_data_points_per_set = data.num_data_points_per_set
        self._dim_outputs = data.dim_outputs
        self._standard_deviation_noise = data.std_noise
        self._num_model_parameters = num_model_parameters
        self._device = device

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = self.log_probs_individual(parameters)
        return sum_individual_log_probs(log_probs, self._num_data_sets)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        model_parameters, model_error_gp_parameters = self._split_parameters(parameters)
        self._model_error_gp.set_parameters(model_error_gp_parameters)
        return self._calculate_log_probs_for_data_sets(
            model_parameters, self._model_error_gp
        )

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        model_parameters = parameters[: self._num_model_parameters]
        model_error_gp_parameters = parameters[self._num_model_parameters :]
        return model_parameters, model_error_gp_parameters

    def _calculate_log_probs_for_data_sets(
        self,
        model_parameters: Tensor,
        model_error_gp: GaussianProcess,
    ) -> Tensor:
        log_probs = tuple(
            self._calculate_one_log_prob_for_data_set(
                model_parameters,
                model_error_gp,
                self._inputs_sets[data_set_index],
                self._outputs_sets[data_set_index],
                self._num_data_points_per_set[data_set_index],
            )
            for data_set_index in range(self._num_data_sets)
        )
        if self._num_data_sets == 1:
            return log_probs[0]
        else:
            return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        num_flattened_outputs = num_data_points * self._dim_outputs
        distribution = self._distribution.initialize(
            model_error_gp, inputs, num_flattened_outputs
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)


class NoiseAndModelErrorGPsOptimizeLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        model_error_gp: Module,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._model_error_gp = model_error_gp.to(device)
        self._distribution = NoiseAndModelErrorGPsLikelihoodDistribution(data, device)
        self._residual_calculator = residual_calculator
        self._inputs_sets = tuple(inputs.detach().to(device) for inputs in data.inputs)
        self._outputs_sets = tuple(
            outputs.detach().to(device) for outputs in data.outputs
        )
        self._num_data_sets = data.num_data_sets
        self._num_data_points_per_set = data.num_data_points_per_set
        self._dim_outputs = data.dim_outputs
        self._standard_deviation_noise = data.std_noise
        self._num_model_parameters = num_model_parameters
        self._device = device

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = self.log_probs_individual(parameters)
        return sum_individual_log_probs(log_probs, self._num_data_sets)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        return self._calculate_log_probs_for_data_sets(
            model_parameters, self._model_error_gp
        )

    def _calculate_log_probs_for_data_sets(
        self,
        model_parameters: Tensor,
        model_error_gp: GaussianProcess,
    ) -> Tensor:
        log_probs = tuple(
            self._calculate_one_log_prob_for_data_set(
                model_parameters,
                model_error_gp,
                self._inputs_sets[data_set_index],
                self._outputs_sets[data_set_index],
                self._num_data_points_per_set[data_set_index],
            )
            for data_set_index in range(self._num_data_sets)
        )
        if self._num_data_sets == 1:
            return log_probs[0]
        else:
            return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        num_flattened_outputs = num_data_points * self._dim_outputs
        distribution = self._distribution.initialize(
            model_error_gp, inputs, num_flattened_outputs
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)


def sum_individual_log_probs(log_probs: Tensor, num_data_sets: int) -> Tensor:
    if num_data_sets == 1:
        return log_probs[0]
    return torch.sum(log_probs)
