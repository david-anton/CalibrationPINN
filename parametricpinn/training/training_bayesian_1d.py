from dataclasses import dataclass
from time import perf_counter
from typing import NamedTuple

import torch

from parametricpinn.ansatz import BayesianAnsatz
from parametricpinn.calibration import (
    EfficientNUTSConfig,
    MCMC_Algorithm_Output,
    calibrate,
)
from parametricpinn.calibration.bayesian.distributions import (
    IndependentMultivariateNormalDistributon,
    create_independent_multivariate_normal_distribution,
)
from parametricpinn.data import (
    TrainingData1DPDE,
    TrainingData1DStressBC,
    TrainingDataset1D,
    collate_training_data_1D,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import ParameterPriorStds
from parametricpinn.training.loss_1d import momentum_equation_func, traction_func
from parametricpinn.types import Device, Tensor


class MeasurementsStds(NamedTuple):
    pde: float
    stress_bc: float


@dataclass
class TrainingConfiguration:
    ansatz: BayesianAnsatz
    parameter_prior_stds: ParameterPriorStds
    training_dataset: TrainingDataset1D
    measurements_standard_deviations: MeasurementsStds
    number_mcmc_iterations: int
    output_subdirectory: str
    project_directory: ProjectDirectory
    device: Device


class TrainigLikelihood:
    def __init__(
        self,
        ansatz: BayesianAnsatz,
        training_dataset: TrainingDataset1D,
        measurements_standard_deviations: MeasurementsStds,
        device: Device,
    ) -> None:
        self._ansatz = ansatz
        self._device = device
        data_pde, data_stress_bc = self._unpack_training_dataset(training_dataset)
        self._pde_data = data_pde
        self._stress_bc_data = data_stress_bc
        self._num_flattened_y_pde = torch.numel(self._pde_data.y_true)
        self._num_flattened_y_stress_bc = torch.numel(self._stress_bc_data.y_true)
        self._flattened_true_outputs = self._assemble_flattened_true_outputs()
        self._measurements_stds = measurements_standard_deviations
        self._likelihood = self._initialize_likelihood()

    def prob(self, parameters: Tensor) -> Tensor:
        self._set_model_parameters(parameters)
        with torch.no_grad():
            return self._prob()

    def log_prob(self, parameters: Tensor) -> Tensor:
        self._set_model_parameters(parameters)
        with torch.no_grad():
            return self._log_prob()

    def grad_log_prob(self, parameters: Tensor) -> Tensor:
        self._set_model_parameters(parameters)
        self._ansatz.network.zero_grad()
        self._log_prob().backward()
        return self._ansatz.network.get_flattened_gradients()

    def _set_model_parameters(self, parameters: Tensor) -> None:
        self._ansatz.network.set_flattened_parameters(parameters)
        self._ansatz.to(self._device)

    def _prob(self) -> Tensor:
        return torch.exp(self._log_prob())

    def _log_prob(self) -> Tensor:
        residual = self._calculate_residuals()
        return self._likelihood.log_prob(residual)

    def _calculate_residuals(self) -> Tensor:
        flattened_ansatz_outputs = self._calculate_flattened_ansatz_outputs()
        return flattened_ansatz_outputs - self._flattened_true_outputs

    def _calculate_flattened_ansatz_outputs(self) -> Tensor:
        def y_pde_func() -> Tensor:
            x_coor = self._pde_data.x_coor.to(self._device)
            x_E = self._pde_data.x_E.to(self._device)
            volume_force = self._pde_data.f.to(self._device)
            return momentum_equation_func(self._ansatz, x_coor, x_E, volume_force)

        def y_stress_bc_func() -> Tensor:
            x_coor = self._stress_bc_data.x_coor.to(self._device)
            x_E = self._stress_bc_data.x_E.to(self._device)
            return traction_func(self._ansatz, x_coor, x_E)

        y_pde = y_pde_func().ravel()
        y_stress_bc = y_stress_bc_func().ravel()
        return torch.concat([y_pde, y_stress_bc], dim=0)

    def _unpack_training_dataset(
        self, training_dataset: TrainingDataset1D
    ) -> tuple[TrainingData1DPDE, TrainingData1DStressBC]:
        training_data = [
            (sample_pde, sample_stress_bc)
            for sample_pde, sample_stress_bc in iter(training_dataset)
        ]
        data_pde, data_stress_bc = collate_training_data_1D(training_data)
        return data_pde, data_stress_bc

    def _assemble_flattened_true_outputs(self) -> Tensor:
        y_pde_true = self._pde_data.y_true.ravel()
        y_stress_bc_true = self._stress_bc_data.y_true.ravel()
        return torch.concat([y_pde_true, y_stress_bc_true], dim=0).to(self._device)

    def _initialize_likelihood(self) -> IndependentMultivariateNormalDistributon:
        means = self._assemble_residual_means()
        standard_deviations = self._assemble_residual_standard_deviations()
        return create_independent_multivariate_normal_distribution(
            means, standard_deviations, self._device
        )

    def _assemble_residual_means(self) -> Tensor:
        means_pde = torch.zeros((self._num_flattened_y_pde,))
        means_stress_bc = torch.zeros((self._num_flattened_y_stress_bc,))
        means = [means_pde, means_stress_bc]
        return torch.concat(means, dim=0).to(self._device)

    def _assemble_residual_standard_deviations(self) -> Tensor:
        stds_pde = torch.full((self._num_flattened_y_pde,), self._measurements_stds.pde)
        stds_stress_bc = torch.full(
            (self._num_flattened_y_stress_bc,),
            self._measurements_stds.stress_bc,
        )
        stds = [stds_pde, stds_stress_bc]
        return torch.concat(stds, dim=0).to(self._device)


def train_parametric_pinn(
    train_config: TrainingConfiguration,
) -> MCMC_Algorithm_Output:
    ansatz = train_config.ansatz
    parameter_prior_stds = train_config.parameter_prior_stds
    training_dataset = train_config.training_dataset
    measurements_standard_deviations = train_config.measurements_standard_deviations
    number_mcmc_iterations = train_config.number_mcmc_iterations
    output_subdir = train_config.output_subdirectory
    project_directory = train_config.project_directory
    device = train_config.device

    parameters_shape = ansatz.network.get_flattened_parameters().shape

    likelihood = TrainigLikelihood(
        ansatz,
        training_dataset,
        measurements_standard_deviations,
        device,
    )

    prior = ansatz.network.create_independent_multivariate_normal_prior(
        parameter_prior_stds, device
    )

    initial_parameters = prior.sample().to(device)
    leapfrog_step_sizes = torch.full(parameters_shape, 1e-6, device=device)

    mcmc_config = EfficientNUTSConfig(
        initial_parameters=initial_parameters,
        num_iterations=number_mcmc_iterations,
        num_burn_in_iterations=int(1e4),
        leapfrog_step_sizes=leapfrog_step_sizes,
        max_tree_depth=10,  # = 1024 steps
    )

    start = perf_counter()
    posterior_moments, samples = calibrate(likelihood, prior, mcmc_config, device)
    end = perf_counter()
    time = end - start
    print(f"Sampling time: {time}")

    return posterior_moments, samples
