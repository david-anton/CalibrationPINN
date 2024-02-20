from typing import Protocol, TypeAlias, Union

import torch
from torch.func import jacrev, vmap

from parametricpinn.ansatz import BayesianAnsatz, StandardAnsatz
from parametricpinn.calibration.data import (
    CalibrationData,
    PreprocessedCalibrationData,
    preprocess_calibration_data,
)
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.gps import GaussianProcess
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


class StandardResidualCalculator:
    def __init__(
        self,
        model: StandardAnsatz,
        data: PreprocessedCalibrationData,
        device: Device,
    ):
        self._model = model.to(device)
        freeze_model(self._model)
        self._data = data
        self._data.inputs.detach().to(device)
        self._data.outputs.detach().to(device)
        self._flattened_data_outputs = self._data.outputs.ravel()
        self._num_flattened_data_outputs = len(self._flattened_data_outputs)
        self._device = device

    def calculate_residuals(self, parameters: Tensor) -> Tensor:
        flattened_model_outputs = self._calculate_flattened_model_outputs(parameters)
        return flattened_model_outputs - self._flattened_data_outputs

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

    def log_probs_pointwise(self, parameters: Tensor) -> Tensor:
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


class NoiseQLikelihoodStrategy:
    def __init__(
        self,
        standard_likelihood_strategy: NoiseLikelihoodStrategy,
        data: PreprocessedCalibrationData,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        self._standard_likelihood_strategy = standard_likelihood_strategy
        self._standard_deviation_noise = data.std_noise
        self._num_data_points = data.num_data_points
        self._num_flattened_outputs = data.num_data_points * data.dim_outputs
        self._num_model_parameters = num_model_parameters
        self._device = device

    def log_prob(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters[: self._num_model_parameters]
        return self._calibrated_log_likelihood(model_parameters)

    def _calibrated_log_likelihood(self, parameters: Tensor) -> Tensor:
        Q = self._determine_q(parameters)
        M = torch.absolute(Q)
        return torch.log(M ** (-1 / 2)) - Q

    def _determine_q(self, parameters: Tensor) -> Tensor:
        scores, total_score = self._calculate_scores(parameters)
        W = self._estimate_covariance(scores, total_score)
        total_score = torch.unsqueeze(total_score, dim=0)
        sqrt_num_data_points = torch.sqrt(
            torch.tensor(self._num_data_points, device=self._device)
        )
        return (1 / 2) * (
            (total_score / sqrt_num_data_points)
            @ torch.inverse(W)
            @ torch.transpose(total_score / sqrt_num_data_points, 0, 1)
        )

    def _calculate_scores(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        scores = jacrev(self._standard_likelihood_strategy.log_probs_pointwise)(
            parameters
        )
        total_score = torch.sum(scores, dim=0)
        return scores, total_score

    def _estimate_covariance(self, scores: Tensor, total_score: Tensor) -> Tensor:
        mean_score = total_score / self._num_data_points

        def _vmap_calculate_covariance(score) -> Tensor:
            deviation = torch.unsqueeze(score - mean_score, dim=0)
            return torch.matmul(torch.transpose(deviation, 0, 1), deviation)

        covariances = vmap(_vmap_calculate_covariance)(scores)
        return torch.mean(covariances, dim=0)


class NoiseAndModelErrorLikelihoodStrategy:
    def __init__(
        self,
        residual_calculator: StandardResidualCalculator,
        data: PreprocessedCalibrationData,
        num_model_parameters: int,
        model_error_gp: GaussianProcess,
        device: Device,
    ) -> None:
        self._residual_calculator = residual_calculator
        self._data = data
        self._data.inputs.detach().to(device)
        self._standard_deviation_noise = data.std_noise
        self._num_flattened_outputs = data.num_data_points * data.dim_outputs
        self._num_model_parameters = num_model_parameters
        self._model_error_gp = model_error_gp.to(device)
        self._device = device

    def log_prob(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters[: self._num_model_parameters]
        residuals = self._residual_calculator.calculate_residuals(model_parameters)
        gp_parameters = parameters[self._num_model_parameters :]
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
        self._model_error_gp.set_parameters(gp_parameters)
        inputs = self._data.inputs
        return self._model_error_gp.forward_kernel(inputs, inputs)


class StandardPPINNLikelihood:
    def __init__(
        self,
        likelihood_strategy: LikelihoodStrategy,
        device: Device,
    ) -> None:
        self._likelihood_strategy = likelihood_strategy
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
        return self._likelihood_strategy.log_prob(parameters)


def create_standard_ppinn_likelihood_for_noise(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    residual_calculator = StandardResidualCalculator(
        model=model,
        data=preprocessed_data,
        device=device,
    )
    likelihood_strategy = NoiseLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )


def create_standard_ppinn_q_likelihood_for_noise(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    residual_calculator = StandardResidualCalculator(
        model=model,
        data=preprocessed_data,
        device=device,
    )
    standard_likelihood_strategy = NoiseLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    q_likelihood_strategy = NoiseQLikelihoodStrategy(
        standard_likelihood_strategy=standard_likelihood_strategy,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=q_likelihood_strategy,
        device=device,
    )


def create_standard_ppinn_likelihood_for_noise_and_model_error(
    model: StandardAnsatz,
    num_model_parameters: int,
    model_error_gp: GaussianProcess,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    preprocessed_data = preprocess_calibration_data(data)
    residual_calculator = StandardResidualCalculator(
        model=model,
        data=preprocessed_data,
        device=device,
    )
    likelihood_strategy = NoiseAndModelErrorLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
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
