from typing import TypeAlias

import torch
from torch.func import jacfwd, vmap

from parametricpinn.ansatz import BayesianAnsatz, StandardAnsatz
from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.bayesianinference.likelihoods.likelihoodstrategies import (
    LikelihoodStrategy,
    NoiseAndModelErrorGPsOptimizeLikelihoodStrategy,
    NoiseAndModelErrorGPsSamplingLikelihoodStrategy,
    NoiseAndModelErrorOptimizeLikelihoodStrategy,
    NoiseAndModelErrorSamplingLikelihoodStrategy,
    NoiseLikelihoodStrategy,
)
from parametricpinn.calibration.bayesianinference.likelihoods.optimization import (
    optimize_hyperparameters,
)
from parametricpinn.calibration.bayesianinference.likelihoods.residualcalculator import (
    StandardResidualCalculator,
)
from parametricpinn.calibration.data import (
    CalibrationData,
    PreprocessedCalibrationData,
    preprocess_calibration_data,
)
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.gps import GaussianProcess
from parametricpinn.statistics.distributions import (
    IndependentMultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
)
from parametricpinn.types import Device, Tensor

QLikelihoodStrategies: TypeAlias = (
    NoiseLikelihoodStrategy
    | NoiseAndModelErrorSamplingLikelihoodStrategy
    | NoiseAndModelErrorOptimizeLikelihoodStrategy
)


class StandardPPINNQLikelihoodWrapper:
    def __init__(
        self,
        likelihood_strategy: QLikelihoodStrategies,
        data: PreprocessedCalibrationData,
        num_model_parameters: int,
        device: Device,
    ) -> None:
        self._likelihood_strategy = likelihood_strategy
        self._standard_deviation_noise = data.std_noise
        self._num_model_parameters = num_model_parameters
        self._device = device

    def log_prob(self, parameters: Tensor) -> Tensor:
        return self._calibrated_log_likelihood(parameters)

    def _calibrated_log_likelihood(self, parameters: Tensor) -> Tensor:
        return self._default_calibrated_log_likelihood(parameters)

    # def _robust_calibrated_log_likelihood(self, parameters: Tensor) -> Tensor:
    #     scores = self._calculate_scores(parameters)
    #     W = self._estimate_robust_covariance_matrix(scores)
    #     Q = self._calculate_q(scores, W)
    #     M = torch.det(W)

    #     return (-1 / 2) * torch.log(M) - Q

    #     # gamma = 0.1
    #     # d_0 = stats.chi2.ppf(0.99, self._num_model_parameters)
    #     # k = (2 * d_0 ** (2 - gamma)) / gamma
    #     # c = d_0**2 - k * d_0**gamma

    #     # def g(x: Tensor) -> Tensor:
    #     #     abs_x = torch.absolute(x)
    #     #     return torch.where(abs_x <= d_0, x**2, k * abs_x**gamma + c)

    #     # return (-1 / 2) * torch.log(M) - (1 / 2) * g(torch.sqrt(2 * Q))

    def _default_calibrated_log_likelihood(self, parameters: Tensor) -> Tensor:
        scores = self._calculate_scores(parameters)
        W = self._estimate_default_covariance_matrix(scores)
        Q = self._calculate_q(scores, W)
        M = torch.det(W)

        return (-1 / 2) * torch.log(M) - Q

    def _calculate_scores(self, parameters: Tensor) -> Tensor:
        return jacfwd(self._likelihood_strategy.flattened_log_probs)(parameters)

    # def _estimate_robust_covariance_matrix(self, scores: Tensor) -> Tensor:
    #     num_scores = len(scores)
    #     mean_score = torch.mean(scores, dim=0)

    #     ### 1
    #     S = scores - mean_score
    #     D_bar = torch.diag(torch.sqrt(torch.mean(S**2, dim=0)))
    #     _, R_bar = torch.linalg.qr(
    #         (S @ torch.inverse(D_bar))
    #         / torch.sqrt(torch.tensor(num_scores - 1, device=self._device))
    #     )
    #     R_bar_T = torch.transpose(R_bar, 0, 1)
    #     covar_bar_inv = torch.inverse(D_bar @ R_bar_T @ R_bar @ D_bar)

    #     ### 2
    #     def vmap_calculate_mahalanobis_distance(score: Tensor) -> Tensor:
    #         deviation = score - mean_score
    #         return deviation @ covar_bar_inv @ deviation

    #     m = vmap(vmap_calculate_mahalanobis_distance)(scores)

    #     ### 3
    #     num_statistics = torch.tensor(self._num_model_parameters, device=self._device)
    #     m_0 = torch.sqrt(num_statistics) + torch.sqrt(
    #         torch.tensor(2, device=self._device)
    #     )

    #     weights = torch.where(
    #         m > m_0,
    #         (m_0 / m) * torch.exp(-((m - m_0) ** 2) / 2),
    #         torch.tensor(1.0, device=self._device),
    #     )

    #     ### 4
    #     redefined_mean_score = (weights @ scores) / torch.sum(weights)
    #     redefined_S = scores - redefined_mean_score

    #     ### 5
    #     D = torch.diag(torch.sqrt(torch.mean(redefined_S**2, dim=0)))
    #     W = torch.diag(weights)

    #     # It is not clear whether the -1 in the square root is inside or outside the sum
    #     _, R = torch.linalg.qr(
    #         (
    #             W
    #             @ redefined_S
    #             @ torch.inverse(D)
    #             / torch.sqrt(
    #                 (torch.sum(weights**2) - torch.tensor(1.0, device=self._device))
    #             )
    #         )
    #     )

    #     ### 6
    #     R_T = torch.transpose(R, 0, 1)
    #     covariance = D @ R_T @ R @ D

    #     return covariance

    def _estimate_default_covariance_matrix(self, scores: Tensor) -> Tensor:
        mean_score = torch.mean(scores, dim=0)

        def _vmap_calculate_covariance(score) -> Tensor:
            deviation = torch.unsqueeze(score - mean_score, dim=0)
            return torch.matmul(torch.transpose(deviation, 0, 1), deviation)

        covariances = vmap(_vmap_calculate_covariance)(scores)
        return torch.mean(covariances, dim=0)

    def _calculate_q(self, scores: Tensor, covariance: Tensor) -> Tensor:
        num_scores = torch.tensor(len(scores), device=self._device)
        sqrt_num_scores = torch.sqrt(num_scores)
        total_score = torch.sum(scores, dim=0)
        scaled_total_score = total_score / sqrt_num_scores
        return torch.squeeze(
            (1 / 2)
            * (scaled_total_score @ torch.inverse(covariance) @ scaled_total_score),
            dim=0,
        )


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


### Noise


def _create_noise_likelihood_strategy(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> tuple[NoiseLikelihoodStrategy, PreprocessedCalibrationData]:
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
    return likelihood_strategy, preprocessed_data


def create_standard_ppinn_likelihood_for_noise(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    likelihood_strategy, _ = _create_noise_likelihood_strategy(
        model=model, num_model_parameters=num_model_parameters, data=data, device=device
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
    likelihood_strategy, preprocessed_data = _create_noise_likelihood_strategy(
        model=model, num_model_parameters=num_model_parameters, data=data, device=device
    )
    q_likelihood_strategy = StandardPPINNQLikelihoodWrapper(
        likelihood_strategy=likelihood_strategy,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=q_likelihood_strategy,
        device=device,
    )


### Noise and model error


def _create_noise_and_model_error_likelihood_strategy_for_sampling(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> tuple[NoiseAndModelErrorSamplingLikelihoodStrategy, PreprocessedCalibrationData]:
    preprocessed_data = preprocess_calibration_data(data)
    residual_calculator = StandardResidualCalculator(
        model=model,
        data=preprocessed_data,
        device=device,
    )
    likelihood_strategy = NoiseAndModelErrorSamplingLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    return likelihood_strategy, preprocessed_data


def _create_optimized_noise_and_model_error_likelihood_strategy(
    model: StandardAnsatz,
    num_model_parameters: int,
    initial_model_error_standard_deviations: Tensor,
    data: CalibrationData,
    prior_material_parameters: Prior,
    num_material_parameter_samples: int,
    num_iterations: int,
    device: Device,
) -> tuple[NoiseAndModelErrorOptimizeLikelihoodStrategy, PreprocessedCalibrationData]:
    preprocessed_data = preprocess_calibration_data(data)
    residual_calculator = StandardResidualCalculator(
        model=model,
        data=preprocessed_data,
        device=device,
    )
    likelihood_strategy = NoiseAndModelErrorOptimizeLikelihoodStrategy(
        initial_model_error_standard_deviations=initial_model_error_standard_deviations,
        residual_calculator=residual_calculator,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    likelihood_strategy.train()
    optimize_hyperparameters(
        likelihood=likelihood_strategy,
        prior_material_parameters=prior_material_parameters,
        num_material_parameter_samples=num_material_parameter_samples,
        num_iterations=num_iterations,
        device=device,
    )
    freeze_model(likelihood_strategy)
    return likelihood_strategy, preprocessed_data


def create_standard_ppinn_likelihood_for_noise_and_model_error_sampling(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    (
        likelihood_strategy,
        _,
    ) = _create_noise_and_model_error_likelihood_strategy_for_sampling(
        model=model,
        num_model_parameters=num_model_parameters,
        data=data,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )


def create_optimized_standard_ppinn_likelihood_for_noise_and_model_error(
    model: StandardAnsatz,
    num_model_parameters: int,
    initial_model_error_standard_deviations: Tensor,
    data: CalibrationData,
    prior_material_parameters: Prior,
    num_material_parameter_samples: int,
    num_iterations: int,
    device: Device,
) -> StandardPPINNLikelihood:
    (
        likelihood_strategy,
        _,
    ) = _create_optimized_noise_and_model_error_likelihood_strategy(
        model=model,
        num_model_parameters=num_model_parameters,
        initial_model_error_standard_deviations=initial_model_error_standard_deviations,
        data=data,
        prior_material_parameters=prior_material_parameters,
        num_material_parameter_samples=num_material_parameter_samples,
        num_iterations=num_iterations,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )


def create_standard_ppinn_q_likelihood_for_noise_and_model_error_sampling(
    model: StandardAnsatz,
    num_model_parameters: int,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    (
        likelihood_strategy,
        preprocessed_data,
    ) = _create_noise_and_model_error_likelihood_strategy_for_sampling(
        model=model, num_model_parameters=num_model_parameters, data=data, device=device
    )
    q_likelihood_strategy = StandardPPINNQLikelihoodWrapper(
        likelihood_strategy=likelihood_strategy,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=q_likelihood_strategy,
        device=device,
    )


def create_optimized_standard_ppinn_q_likelihood_for_noise_and_model_error(
    model: StandardAnsatz,
    num_model_parameters: int,
    initial_model_error_standard_deviations: Tensor,
    data: CalibrationData,
    prior_material_parameters: Prior,
    num_material_parameter_samples: int,
    num_iterations: int,
    device: Device,
) -> StandardPPINNLikelihood:
    (
        likelihood_strategy,
        preprocessed_data,
    ) = _create_optimized_noise_and_model_error_likelihood_strategy(
        model=model,
        num_model_parameters=num_model_parameters,
        initial_model_error_standard_deviations=initial_model_error_standard_deviations,
        data=data,
        prior_material_parameters=prior_material_parameters,
        num_material_parameter_samples=num_material_parameter_samples,
        num_iterations=num_iterations,
        device=device,
    )
    q_likelihood_strategy = StandardPPINNQLikelihoodWrapper(
        likelihood_strategy=likelihood_strategy,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=q_likelihood_strategy,
        device=device,
    )


### Noise and model error GPs


def _create_noise_and_model_error_gps_likelihood_strategy_for_sampling(
    model: StandardAnsatz,
    num_model_parameters: int,
    model_error_gp: GaussianProcess,
    data: CalibrationData,
    device: Device,
) -> tuple[
    NoiseAndModelErrorGPsSamplingLikelihoodStrategy, PreprocessedCalibrationData
]:
    preprocessed_data = preprocess_calibration_data(data)
    residual_calculator = StandardResidualCalculator(
        model=model,
        data=preprocessed_data,
        device=device,
    )
    likelihood_strategy = NoiseAndModelErrorGPsSamplingLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=preprocessed_data,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        device=device,
    )
    return likelihood_strategy, preprocessed_data


def _create_optimized_noise_and_model_error_gps_likelihood_strategy(
    model: StandardAnsatz,
    num_model_parameters: int,
    model_error_gp: GaussianProcess,
    initial_model_error_gp_parameters: Tensor,
    data: CalibrationData,
    prior_material_parameters: Prior,
    num_material_parameter_samples: int,
    num_iterations: int,
    device: Device,
) -> tuple[
    NoiseAndModelErrorGPsOptimizeLikelihoodStrategy, PreprocessedCalibrationData
]:
    preprocessed_data = preprocess_calibration_data(data)
    residual_calculator = StandardResidualCalculator(
        model=model,
        data=preprocessed_data,
        device=device,
    )
    likelihood_strategy = NoiseAndModelErrorGPsOptimizeLikelihoodStrategy(
        model_error_gp=model_error_gp,
        initial_model_error_gp_parameters=initial_model_error_gp_parameters,
        data=preprocessed_data,
        residual_calculator=residual_calculator,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    likelihood_strategy.train()
    optimize_hyperparameters(
        likelihood=likelihood_strategy,
        prior_material_parameters=prior_material_parameters,
        num_material_parameter_samples=num_material_parameter_samples,
        num_iterations=num_iterations,
        device=device,
    )
    freeze_model(likelihood_strategy)
    return likelihood_strategy, preprocessed_data


def create_standard_ppinn_likelihood_for_noise_and_model_error_gps_sampling(
    model: StandardAnsatz,
    num_model_parameters: int,
    model_error_gp: GaussianProcess,
    data: CalibrationData,
    device: Device,
) -> StandardPPINNLikelihood:
    (
        likelihood_strategy,
        _,
    ) = _create_noise_and_model_error_gps_likelihood_strategy_for_sampling(
        model=model,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        data=data,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )


def create_optimized_standard_ppinn_likelihood_for_noise_and_model_error_gps(
    model: StandardAnsatz,
    num_model_parameters: int,
    model_error_gp: GaussianProcess,
    initial_model_error_gp_parameters: Tensor,
    data: CalibrationData,
    prior_material_parameters: Prior,
    num_material_parameter_samples: int,
    num_iterations: int,
    device: Device,
) -> StandardPPINNLikelihood:
    (
        likelihood_strategy,
        _,
    ) = _create_optimized_noise_and_model_error_gps_likelihood_strategy(
        model=model,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        initial_model_error_gp_parameters=initial_model_error_gp_parameters,
        data=data,
        prior_material_parameters=prior_material_parameters,
        num_material_parameter_samples=num_material_parameter_samples,
        num_iterations=num_iterations,
        device=device,
    )
    return StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )


##### Bayesian PPINN likelihoods


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
