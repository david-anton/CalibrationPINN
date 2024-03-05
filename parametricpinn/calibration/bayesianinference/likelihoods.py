from typing import Protocol, TypeAlias, Union

import torch
from scipy import stats
from torch.func import jacfwd, vmap

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
        make_robust: bool,
        device: Device,
    ) -> None:
        self._standard_likelihood_strategy = standard_likelihood_strategy
        self._standard_deviation_noise = data.std_noise
        self._num_flattened_data_points = data.num_data_points * data.dim_outputs
        self._num_model_parameters = num_model_parameters
        self._make_robust = make_robust
        self._device = device

    def log_prob(self, parameters: Tensor) -> Tensor:
        model_parameters = parameters[: self._num_model_parameters]
        return self._calibrated_log_likelihood(model_parameters)

    def _calibrated_log_likelihood(self, parameters: Tensor) -> Tensor:
        if self._make_robust:
            return self._robust_calibrated_log_likelihood(parameters)
        return self._default_calibrated_log_likelihood(parameters)

    def _robust_calibrated_log_likelihood(self, parameters: Tensor) -> Tensor:
        scores, total_score = self._calculate_scores(parameters)
        W = self._estimate_robust_covariance_matrix(scores, total_score)
        Q = self._calculate_q(total_score, W)
        M = torch.det(W)

        gamma = 0.1
        d_0 = stats.chi2.ppf(0.99, self._num_model_parameters)
        k = (2 * d_0 ** (2 - gamma)) / gamma
        c = d_0**2 - k * d_0**gamma

        def g(x: Tensor) -> Tensor:
            abs_x = torch.absolute(x)
            return torch.where(abs_x <= d_0, x**2, k * abs_x**gamma + c)

        return (-1 / 2) * torch.log(M) - (1 / 2) * g(torch.sqrt(2 * Q))

    def _default_calibrated_log_likelihood(self, parameters: Tensor) -> Tensor:
        scores, total_score = self._calculate_scores(parameters)
        W = self._estimate_default_covariance_matrix(scores, total_score)
        Q = self._calculate_q(total_score, W)
        M = torch.det(W)

        return (-1 / 2) * torch.log(M) - Q

    def _calculate_scores(self, parameters: Tensor) -> tuple[Tensor, Tensor]:
        random_indices = torch.randperm(self._num_flattened_data_points)

        def group_log_probs(parameters: Tensor) -> Tensor:
            log_probs = self._standard_likelihood_strategy.log_probs_pointwise(
                parameters
            )
            shuffled_log_probs = log_probs[random_indices]
            return torch.sum(shuffled_log_probs.reshape((-1, 32)), dim=1)

        scores = jacfwd(group_log_probs)(parameters)
        # scores = jacfwd(self._standard_likelihood_strategy.log_probs_pointwise)(
        #     parameters
        # )

        total_score = torch.sum(scores, dim=0)
        return scores, total_score

    def _estimate_robust_covariance_matrix(
        self, scores: Tensor, total_score: Tensor
    ) -> Tensor:
        mean_score = total_score / self._num_flattened_data_points

        ### 1
        S = scores - mean_score
        D_bar = torch.diag(torch.sqrt(torch.mean(S**2, dim=0)))
        _, R_bar = torch.linalg.qr(
            (S @ torch.inverse(D_bar))
            / torch.sqrt(
                torch.tensor(self._num_flattened_data_points - 1, device=self._device)
            )
        )
        R_bar_T = torch.transpose(R_bar, 0, 1)
        covar_bar_inv = torch.inverse(D_bar @ R_bar_T @ R_bar @ D_bar)

        ### 2
        def vmap_calculate_mahalanobis_distance(score: Tensor) -> Tensor:
            deviation = score - mean_score
            return deviation @ covar_bar_inv @ deviation

        m = vmap(vmap_calculate_mahalanobis_distance)(scores)

        ### 3
        num_statistics = torch.tensor(self._num_model_parameters, device=self._device)
        m_0 = torch.sqrt(num_statistics) + torch.sqrt(
            torch.tensor(2, device=self._device)
        )

        weights = torch.where(
            m > m_0,
            (m_0 / m) * torch.exp(-((m - m_0) ** 2) / 2),
            torch.tensor(1.0, device=self._device),
        )

        ### 4
        redefined_mean_score = (weights @ scores) / torch.sum(weights)
        redefined_S = scores - redefined_mean_score

        ### 5
        D = torch.diag(torch.sqrt(torch.mean(redefined_S**2, dim=0)))
        W = torch.diag(weights)

        # It is not clear whether the -1 in the square root is inside or outside the sum
        _, R = torch.linalg.qr(
            (
                W
                @ redefined_S
                @ torch.inverse(D)
                / torch.sqrt(
                    (torch.sum(weights**2) - torch.tensor(1.0, device=self._device))
                )
            )
        )

        ### 6
        R_T = torch.transpose(R, 0, 1)
        covariance = D @ R_T @ R @ D

        return covariance

    def _estimate_default_covariance_matrix(
        self, scores: Tensor, total_score: Tensor
    ) -> Tensor:
        mean_score = total_score / self._num_flattened_data_points

        def _vmap_calculate_covariance(score) -> Tensor:
            deviation = torch.unsqueeze(score - mean_score, dim=0)
            return torch.matmul(torch.transpose(deviation, 0, 1), deviation)

        covariances = vmap(_vmap_calculate_covariance)(scores)
        return torch.mean(covariances, dim=0)

    def _calculate_q(self, total_score: Tensor, covariance: Tensor) -> Tensor:
        sqrt_num_data_points = torch.sqrt(
            torch.tensor(self._num_flattened_data_points, device=self._device)
        )
        return torch.squeeze(
            (1 / 2)
            * (
                (total_score / sqrt_num_data_points)
                @ torch.inverse(covariance)
                @ torch.transpose(
                    torch.unsqueeze((total_score / sqrt_num_data_points), dim=0), 0, 1
                )
            ),
            dim=0,
        )


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
    make_robust: bool,
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
        make_robust=make_robust,
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
