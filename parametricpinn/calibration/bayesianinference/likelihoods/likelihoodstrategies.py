from typing import Optional, Protocol, TypeAlias

import torch

from parametricpinn.calibration.bayesianinference.likelihoods.residualcalculator import (
    StandardResidualCalculator,
)
from parametricpinn.calibration.data import PreprocessedCalibrationData
from parametricpinn.errors import OptimizedModelErrorGPLikelihoodStrategyError
from parametricpinn.gps import GaussianProcess
from parametricpinn.statistics.distributions import (
    IndependentMultivariateNormalDistributon,
    MultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
    create_multivariate_normal_distribution,
)
from parametricpinn.types import Device, Parameter, Tensor

NamedParameters: TypeAlias = dict[str, Tensor]
NamedParametersTuple: TypeAlias = tuple[NamedParameters, ...]
TensorTuple: TypeAlias = tuple[Tensor, ...]
GaussianProcessTuple: TypeAlias = tuple[GaussianProcess, ...]


class LikelihoodStrategy(Protocol):
    def forward(self, parameters: Tensor) -> Tensor:
        pass

    def log_prob(self, parameters: Tensor) -> Tensor:
        pass

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        pass


class OptimizedLikelihoodStrategy(LikelihoodStrategy, Protocol):
    num_data_sets: int

    def log_prob_individual_data_set(
        self, parameters: Tensor, data_set_index: int
    ) -> Tensor:
        pass

    def get_named_parameters(self) -> NamedParametersTuple:
        pass


### Noise
class NoiseLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._noise_standard_deviation = make_sure_it_is_a_tesnor(
            data.std_noise, device
        )
        self._device = device

    def initialize(
        self, num_data_points: int, dim_outputs: int
    ) -> IndependentMultivariateNormalDistributon:
        num_flattened_outputs = num_data_points * dim_outputs
        means = self._assemble_means(num_flattened_outputs)
        standard_deviations = assemble_noise_standard_deviations(
            noise_standard_deviation=self._noise_standard_deviation,
            num_data_points=num_data_points,
            dim_outputs=dim_outputs,
            device=self._device,
        )
        return create_independent_multivariate_normal_distribution(
            means=means,
            standard_deviations=standard_deviations,
            device=self._device,
        )

    def _assemble_means(self, num_flattened_outputs: int) -> Tensor:
        return torch.zeros(
            (num_flattened_outputs,), dtype=torch.float64, device=self._device
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
        return torch.sum(log_probs)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        if self._num_data_sets == 1:
            return self._log_probs_individual_data_points(model_parameters)
        return self._log_probs_individual_data_sets(model_parameters)

    def _log_probs_individual_data_points(self, model_parameters: Tensor) -> Tensor:
        data_set_index = 0
        return self._calculate_log_probs_for_data_points(
            model_parameters,
            self._inputs_sets[data_set_index],
            self._outputs_sets[data_set_index],
            self._num_data_points_per_set[data_set_index],
        )

    def _calculate_log_probs_for_data_points(
        self,
        model_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._initialize_residuals_distribution(num_data_points)
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return distribution.log_probs_individual(residuals)

    def _log_probs_individual_data_sets(self, model_parameters: Tensor) -> Tensor:
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
        return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._initialize_residuals_distribution(num_data_points)
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)

    def _initialize_residuals_distribution(
        self, num_data_points: int
    ) -> IndependentMultivariateNormalDistributon:
        return self._distribution.initialize(num_data_points, self._dim_outputs)


### Noise and model error
class NoiseAndErrorLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._noise_standard_deviation = make_sure_it_is_a_tesnor(
            data.std_noise, device
        )
        self._device = device

    def initialize(
        self,
        model_error_standard_deviation_parameters: Tensor | Parameter,
        num_data_points: int,
        dim_outputs: int,
    ) -> IndependentMultivariateNormalDistributon:
        num_flattened_outputs = num_data_points * dim_outputs
        means = self._assemble_means(num_flattened_outputs)
        standard_deviations = self._assemble_standard_deviations(
            model_error_standard_deviation_parameters,
            num_data_points,
            dim_outputs,
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
        dim_outputs: int,
    ) -> Tensor:
        noise_standard_deviations = assemble_noise_standard_deviations(
            noise_standard_deviation=self._noise_standard_deviation,
            num_data_points=num_data_points,
            dim_outputs=dim_outputs,
            device=self._device,
        )
        model_error_standard_deviations = self._assemble_error_standard_deviations(
            model_error_standard_deviation_parameters, num_data_points
        )
        return torch.sqrt(
            noise_standard_deviations**2 + model_error_standard_deviations**2
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


class NoiseAndErrorSamplingLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self._distribution = NoiseAndErrorLikelihoodDistribution(data, device)
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
        return torch.sum(log_probs)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        (
            model_parameters,
            model_error_standard_deviation_parameters,
        ) = self._split_parameters(parameters)
        if self._num_data_sets == 1:
            return self._log_probs_individual_data_points(
                model_parameters, model_error_standard_deviation_parameters
            )
        return self._log_probs_individual_data_sets(
            model_parameters, model_error_standard_deviation_parameters
        )

    def _split_parameters(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        model_parameters = parameters[: self._num_model_parameters]
        model_error_standard_deviation_parameters = parameters[
            self._num_model_parameters :
        ]
        return model_parameters, model_error_standard_deviation_parameters

    def _log_probs_individual_data_points(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
    ) -> Tensor:
        data_set_index = 0
        return self._calculate_log_probs_for_data_points(
            model_parameters,
            model_error_standard_deviation_parameters,
            self._inputs_sets[data_set_index],
            self._outputs_sets[data_set_index],
            self._num_data_points_per_set[data_set_index],
        )

    def _calculate_log_probs_for_data_points(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._initialize_residuals_distribution(
            model_error_standard_deviation_parameters,
            num_data_points,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return distribution.log_probs_individual(residuals)

    def _log_probs_individual_data_sets(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
    ) -> Tensor:
        return self._calculate_log_probs_for_data_sets(
            model_parameters, model_error_standard_deviation_parameters
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
        return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._initialize_residuals_distribution(
            model_error_standard_deviation_parameters,
            num_data_points,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)

    def _initialize_residuals_distribution(
        self, model_error_standard_deviation_parameters: Tensor, num_data_points: int
    ) -> IndependentMultivariateNormalDistributon:
        return self._distribution.initialize(
            model_error_standard_deviation_parameters,
            num_data_points,
            self._dim_outputs,
        )


class NoiseAndErrorOptimizedLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        initial_model_error_standard_deviations: TensorTuple,
        use_independent_model_error_standard_deviations: bool,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self.num_data_sets = data.num_data_sets
        self._model_error_standard_deviation_parameters = torch.nn.ParameterList(
            torch.nn.Parameter(standard_deviations.clone(), requires_grad=True)
            for standard_deviations in initial_model_error_standard_deviations
        )
        self._use_independent_model_error_standard_deviations = (
            use_independent_model_error_standard_deviations
        )
        self._distribution = NoiseAndErrorLikelihoodDistribution(data, device)
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
        return torch.sum(log_probs)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        if self._num_data_sets == 1:
            return self._log_probs_individual_data_points(model_parameters)
        return self._log_probs_individual_data_sets(model_parameters)

    def log_prob_individual_data_set(
        self, parameters: Tensor, data_set_index: int
    ) -> Tensor:
        model_parameters = parameters
        if self._use_independent_model_error_standard_deviations:
            log_prob = self._calculate_one_log_prob_for_data_set(
                model_parameters,
                self._model_error_standard_deviation_parameters[data_set_index],
                self._inputs_sets[data_set_index],
                self._outputs_sets[data_set_index],
                self._num_data_points_per_set[data_set_index],
            )
        else:
            log_prob = self._calculate_one_log_prob_for_data_set(
                model_parameters,
                self._model_error_standard_deviation_parameters[0],
                self._inputs_sets[data_set_index],
                self._outputs_sets[data_set_index],
                self._num_data_points_per_set[data_set_index],
            )
        return log_prob[0]

    def get_named_parameters(self) -> NamedParametersTuple:
        def _get_one_named_parameters(data_set_index: int) -> NamedParameters:
            return {
                f"stddev_dimension_{count}": parameter
                for count, parameter in enumerate(
                    self._model_error_standard_deviation_parameters[data_set_index]
                )
            }

        return tuple(
            _get_one_named_parameters(data_set_index)
            for data_set_index in range(self.num_data_sets)
        )

    def _log_probs_individual_data_points(
        self,
        model_parameters: Tensor,
    ) -> Tensor:
        data_set_index = 0
        return self._calculate_log_probs_for_data_points(
            model_parameters,
            self._model_error_standard_deviation_parameters[data_set_index],
            self._inputs_sets[data_set_index],
            self._outputs_sets[data_set_index],
            self._num_data_points_per_set[data_set_index],
        )

    def _calculate_log_probs_for_data_points(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._initialize_residuals_distribution(
            model_error_standard_deviation_parameters,
            num_data_points,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return distribution.log_probs_individual(residuals)

    def _log_probs_individual_data_sets(self, model_parameters: Tensor) -> Tensor:
        return self._calculate_log_probs_for_data_sets(model_parameters)

    def _calculate_log_probs_for_data_sets(
        self,
        model_parameters: Tensor,
    ) -> Tensor:
        if self._use_independent_model_error_standard_deviations:
            log_probs = tuple(
                self._calculate_one_log_prob_for_data_set(
                    model_parameters,
                    self._model_error_standard_deviation_parameters[data_set_index],
                    self._inputs_sets[data_set_index],
                    self._outputs_sets[data_set_index],
                    self._num_data_points_per_set[data_set_index],
                )
                for data_set_index in range(self.num_data_sets)
            )
        else:
            log_probs = tuple(
                self._calculate_one_log_prob_for_data_set(
                    model_parameters,
                    self._model_error_standard_deviation_parameters[0],
                    self._inputs_sets[data_set_index],
                    self._outputs_sets[data_set_index],
                    self._num_data_points_per_set[data_set_index],
                )
                for data_set_index in range(self.num_data_sets)
            )
        return torch.concat(log_probs)

    def _calculate_one_log_prob_for_data_set(
        self,
        model_parameters: Tensor,
        model_error_standard_deviation_parameters: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._initialize_residuals_distribution(
            model_error_standard_deviation_parameters,
            num_data_points,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)

    def _initialize_residuals_distribution(
        self,
        model_error_standard_deviation_parameters: Tensor,
        num_data_points: int,
    ) -> IndependentMultivariateNormalDistributon:
        return self._distribution.initialize(
            model_error_standard_deviation_parameters,
            num_data_points,
            self._dim_outputs,
        )


### Noise and model error GPs
class NoiseAndErrorGPsLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._noise_standard_deviation = make_sure_it_is_a_tesnor(
            data.std_noise, device
        )
        self._device = device

    def initialize(
        self,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        num_data_points: int,
        dim_outputs: int,
    ) -> MultivariateNormalDistributon:
        num_flattened_outputs = num_data_points * dim_outputs
        means = self._assemble_means(
            model_error_gp=model_error_gp,
            inputs=inputs,
            num_flattened_outputs=num_flattened_outputs,
        )
        covariance_matrix = self._assemble_covariance_matrix(
            model_error_gp=model_error_gp,
            inputs=inputs,
            num_data_points=num_data_points,
            dim_outputs=dim_outputs,
        )
        return create_multivariate_normal_distribution(
            means=means, covariance_matrix=covariance_matrix, device=self._device
        )

    def _assemble_means(
        self,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        num_flattened_outputs: int,
    ) -> Tensor:
        noise_means = self._assemble_noise_means(num_flattened_outputs)
        error_means = self._assemble_error_means(model_error_gp, inputs)
        return noise_means + error_means

    def _assemble_noise_means(self, num_flattened_outputs: int) -> Tensor:
        return torch.zeros(
            (num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_error_means(
        self, model_error_gp: GaussianProcess, inputs: Tensor
    ) -> Tensor:
        return model_error_gp.forward_mean(inputs)

    def _assemble_covariance_matrix(
        self,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        num_data_points: int,
        dim_outputs: int,
    ) -> Tensor:
        noise_covariance = self._assemble_noise_covariance_matrix(
            num_data_points, dim_outputs
        )
        error_covariance = self._assemble_error_covariance_matrix(
            model_error_gp, inputs
        )
        return noise_covariance + error_covariance

    def _assemble_noise_covariance_matrix(
        self, num_data_points: int, dim_outputs: int
    ) -> Tensor:
        noise_standard_deviations = assemble_noise_standard_deviations(
            noise_standard_deviation=self._noise_standard_deviation,
            num_data_points=num_data_points,
            dim_outputs=dim_outputs,
            device=self._device,
        )
        return (
            torch.diag(noise_standard_deviations**2)
            .type(torch.float64)
            .to(self._device)
        )

    def _assemble_error_covariance_matrix(
        self, model_error_gp: GaussianProcess, inputs: Tensor
    ) -> Tensor:
        return model_error_gp.forward_kernel(inputs, inputs)


class NoiseAndErrorGPsOptimizedLikelihoodDistribution:
    def __init__(
        self,
        data: PreprocessedCalibrationData,
        device: Device,
    ) -> None:
        self._noise_standard_deviation = make_sure_it_is_a_tesnor(
            data.std_noise, device
        )
        self._device = device

    def initialize(
        self,
        error_mean: Tensor,
        error_covariance: Tensor,
        num_data_points: int,
        dim_outputs: int,
    ) -> MultivariateNormalDistributon:
        num_flattened_outputs = num_data_points * dim_outputs
        means = self._assemble_mean(
            error_mean=error_mean, num_flattened_outputs=num_flattened_outputs
        )
        covariance_matrix = self._assemble_covariance_matrix(
            error_covariance=error_covariance,
            num_data_points=num_data_points,
            dim_outputs=dim_outputs,
        )
        return create_multivariate_normal_distribution(
            means=means, covariance_matrix=covariance_matrix, device=self._device
        )

    def assemble_error_covariance_matrices(
        self, model_error_gp: GaussianProcess, inputs_sets: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        return tuple(
            model_error_gp.forward_kernel(inputs, inputs) for inputs in inputs_sets
        )

    def assemble_error_means(
        self, model_error_gp: GaussianProcess, inputs_sets: tuple[Tensor, ...]
    ) -> tuple[Tensor, ...]:
        return tuple(model_error_gp.forward_mean(inputs) for inputs in inputs_sets)

    def _assemble_mean(self, error_mean: Tensor, num_flattened_outputs: int) -> Tensor:
        noise_means = self._assemble_noise_mean(num_flattened_outputs)
        return noise_means + error_mean

    def _assemble_noise_mean(self, num_flattened_outputs: int) -> Tensor:
        return torch.zeros(
            (num_flattened_outputs,), dtype=torch.float64, device=self._device
        )

    def _assemble_covariance_matrix(
        self,
        error_covariance: Tensor,
        num_data_points: int,
        dim_outputs: int,
    ) -> Tensor:
        noise_covariance = self._assemble_noise_covariance_matrix(
            num_data_points, dim_outputs
        )
        return noise_covariance + error_covariance

    def _assemble_noise_covariance_matrix(
        self, num_data_points: int, dim_outputs: int
    ) -> Tensor:
        noise_standard_deviations = assemble_noise_standard_deviations(
            noise_standard_deviation=self._noise_standard_deviation,
            num_data_points=num_data_points,
            dim_outputs=dim_outputs,
            device=self._device,
        )
        return (
            torch.diag(noise_standard_deviations**2)
            .type(torch.float64)
            .to(self._device)
        )


class NoiseAndErrorGPsSamplingLikelihoodStrategy(torch.nn.Module):
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
        self._distribution = NoiseAndErrorGPsLikelihoodDistribution(data, device)
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
        return torch.sum(log_probs)

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
        distribution = self._distribution.initialize(
            model_error_gp=model_error_gp,
            inputs=inputs,
            num_data_points=num_data_points,
            dim_outputs=self._dim_outputs,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)


class NoiseAndErrorGPsOptimizedLikelihoodStrategy(torch.nn.Module):
    def __init__(
        self,
        model_error_gps: GaussianProcessTuple,
        use_independent_model_error_gps: bool,
        data: PreprocessedCalibrationData,
        residual_calculator: StandardResidualCalculator,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        super().__init__()
        self.num_data_sets = data.num_data_sets
        self._validate_number_of_model_error_gps(
            model_error_gps, use_independent_model_error_gps, data
        )
        self._model_error_gps = torch.nn.ModuleList(
            tuple(gp.to(device) for gp in model_error_gps)
        )
        self._use_independent_gps = use_independent_model_error_gps
        self._is_training_mode = True
        self._distribution_training = NoiseAndErrorGPsLikelihoodDistribution(
            data, device
        )
        self._error_means_sets: Optional[tuple[Tensor, ...]] = None
        self._error_covariances_sets: Optional[tuple[Tensor, ...]] = None
        self._distribution_prediction = NoiseAndErrorGPsOptimizedLikelihoodDistribution(
            data, device
        )
        self._residual_calculator = residual_calculator
        self._inputs_sets = tuple(inputs.detach().to(device) for inputs in data.inputs)
        self._outputs_sets = tuple(
            outputs.detach().to(device) for outputs in data.outputs
        )
        self._num_data_points_per_set = data.num_data_points_per_set
        self._dim_outputs = data.dim_outputs
        self._standard_deviation_noise = data.std_noise
        self._num_model_parameters = num_model_parameters
        self._device = device

    def training_mode(self) -> None:
        self._is_training_mode = True
        self._error_means_sets = None
        self._error_covariances_sets = None

    def prediction_mode(self) -> None:
        self._is_training_mode = False
        self._error_means_sets = self._assemble_error_means_for_prediction()
        self._error_covariances_sets = (
            self._assemble_error_covariance_matrices_for_prediction()
        )

    def forward(self, parameters: Tensor) -> Tensor:
        return self.log_prob(parameters)

    def log_prob(self, parameters: Tensor) -> Tensor:
        log_probs = self.log_probs_individual(parameters)
        return torch.sum(log_probs)

    def log_probs_individual(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters
        return self._calculate_log_probs_for_data_sets(model_parameters)

    def log_prob_individual_data_set(
        self, parameters: Tensor, data_set_index: int
    ) -> Tensor:
        model_parameters = parameters
        if self._is_training_mode:
            if self._use_independent_gps:
                log_prob = self._calculate_one_log_prob_for_data_set_in_training(
                    model_parameters,
                    self._model_error_gps[data_set_index],
                    self._inputs_sets[data_set_index],
                    self._outputs_sets[data_set_index],
                    self._num_data_points_per_set[data_set_index],
                )
            else:
                log_prob = self._calculate_one_log_prob_for_data_set_in_training(
                    model_parameters,
                    self._model_error_gps[0],
                    self._inputs_sets[data_set_index],
                    self._outputs_sets[data_set_index],
                    self._num_data_points_per_set[data_set_index],
                )
        else:
            if (
                self._error_covariances_sets is not None
                and self._error_means_sets is not None
            ):
                log_prob = self._calculate_one_log_prob_for_data_set_in_prediction(
                    model_parameters,
                    self._error_means_sets[data_set_index],
                    self._error_covariances_sets[data_set_index],
                    self._inputs_sets[data_set_index],
                    self._outputs_sets[data_set_index],
                    self._num_data_points_per_set[data_set_index],
                )

            else:
                raise OptimizedModelErrorGPLikelihoodStrategyError(
                    "The means and covariance matrices of the model errors \
                    must first be calculated for the prediction."
                )
        return log_prob[0]

    def get_named_parameters(self) -> NamedParametersTuple:
        return tuple(gp.get_named_parameters() for gp in self._model_error_gps)

    def _calculate_log_probs_for_data_sets(
        self,
        model_parameters: Tensor,
    ) -> Tensor:
        if self._is_training_mode:
            if self._use_independent_gps:
                log_probs = tuple(
                    self._calculate_one_log_prob_for_data_set_in_training(
                        model_parameters,
                        self._model_error_gps[data_set_index],
                        self._inputs_sets[data_set_index],
                        self._outputs_sets[data_set_index],
                        self._num_data_points_per_set[data_set_index],
                    )
                    for data_set_index in range(self.num_data_sets)
                )
            else:
                log_probs = tuple(
                    self._calculate_one_log_prob_for_data_set_in_training(
                        model_parameters,
                        self._model_error_gps[0],
                        self._inputs_sets[data_set_index],
                        self._outputs_sets[data_set_index],
                        self._num_data_points_per_set[data_set_index],
                    )
                    for data_set_index in range(self.num_data_sets)
                )
            if self.num_data_sets == 1:
                return log_probs[0]
            else:
                return torch.concat(log_probs)
        else:
            if (
                self._error_covariances_sets is not None
                and self._error_means_sets is not None
            ):
                log_probs = tuple(
                    self._calculate_one_log_prob_for_data_set_in_prediction(
                        model_parameters,
                        self._error_means_sets[data_set_index],
                        self._error_covariances_sets[data_set_index],
                        self._inputs_sets[data_set_index],
                        self._outputs_sets[data_set_index],
                        self._num_data_points_per_set[data_set_index],
                    )
                    for data_set_index in range(self.num_data_sets)
                )
                if self.num_data_sets == 1:
                    return log_probs[0]
                else:
                    return torch.concat(log_probs)
            else:
                raise OptimizedModelErrorGPLikelihoodStrategyError(
                    "The means and covariance matrices of the model errors \
                    must first be calculated for the prediction."
                )

    def _calculate_one_log_prob_for_data_set_in_training(
        self,
        model_parameters: Tensor,
        model_error_gp: GaussianProcess,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._distribution_training.initialize(
            model_error_gp=model_error_gp,
            inputs=inputs,
            num_data_points=num_data_points,
            dim_outputs=self._dim_outputs,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)

    def _calculate_one_log_prob_for_data_set_in_prediction(
        self,
        model_parameters: Tensor,
        error_mean: Tensor,
        error_covariance: Tensor,
        inputs: Tensor,
        outputs: Tensor,
        num_data_points: int,
    ) -> Tensor:
        distribution = self._distribution_prediction.initialize(
            error_mean=error_mean,
            error_covariance=error_covariance,
            num_data_points=num_data_points,
            dim_outputs=self._dim_outputs,
        )
        residuals = self._residual_calculator.calculate_residuals(
            model_parameters, inputs, outputs
        )
        return torch.unsqueeze(distribution.log_prob(residuals), dim=0)

    def _validate_number_of_model_error_gps(
        self,
        model_error_gps: GaussianProcessTuple,
        use_independent_model_error_gps: bool,
        data: PreprocessedCalibrationData,
    ) -> None:
        num_model_error_gps = len(model_error_gps)
        num_data_sets = data.num_data_sets
        if use_independent_model_error_gps:
            if not num_model_error_gps == num_data_sets:
                raise OptimizedModelErrorGPLikelihoodStrategyError(
                    f"The number of independent model error GPs {num_model_error_gps} \
                    does not match the number of data sets {num_data_sets}."
                )
        else:
            if not num_model_error_gps == 1:
                raise OptimizedModelErrorGPLikelihoodStrategyError(
                    f"The number of not independent model error GPs {num_model_error_gps} \
                    is expected to be 1."
                )

    def _assemble_error_means_for_prediction(self) -> tuple[Tensor, ...]:
        if self._use_independent_gps:
            return tuple(
                self._distribution_prediction.assemble_error_means(gp, (inputs_set,))[0]
                for gp, inputs_set in zip(self._model_error_gps, self._inputs_sets)
            )
        else:
            return self._distribution_prediction.assemble_error_means(
                self._model_error_gps[0], self._inputs_sets
            )

    def _assemble_error_covariance_matrices_for_prediction(self) -> tuple[Tensor, ...]:
        if self._use_independent_gps:
            return tuple(
                self._distribution_prediction.assemble_error_covariance_matrices(
                    gp, (inputs_set,)
                )[0]
                for gp, inputs_set in zip(self._model_error_gps, self._inputs_sets)
            )
        else:
            return self._distribution_prediction.assemble_error_covariance_matrices(
                self._model_error_gps[0], self._inputs_sets
            )


def make_sure_it_is_a_tesnor(value: float | Tensor, device: Device) -> Tensor:
    return torch.tensor([value], device=device) if isinstance(value, float) else value


def assemble_noise_standard_deviations(
    noise_standard_deviation: Tensor,
    dim_outputs: int,
    num_data_points: int,
    device: Device,
) -> Tensor:
    num_flattened_outputs = dim_outputs * num_data_points
    if len(noise_standard_deviation) == 1:
        return noise_standard_deviation * torch.ones(
            (num_flattened_outputs,),
            dtype=torch.float64,
            device=device,
        )
    else:
        return (
            torch.transpose(noise_standard_deviation.repeat((num_data_points, 1)), 1, 0)
            .ravel()
            .type(torch.float64)
            .to(device)
        )
