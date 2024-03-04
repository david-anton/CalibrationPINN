import os
from datetime import date
from time import perf_counter

import torch

from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import (
    Prior,
    create_multivariate_uniform_distributed_prior,
)
from parametricpinn.calibration import (
    CalibrationData,
    MetropolisHastingsConfig,
    test_coverage,
)
from parametricpinn.calibration.bayesianinference.likelihoods import (
    create_standard_ppinn_likelihood_for_noise,
    create_standard_ppinn_q_likelihood_for_noise,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.types import NPArray, Tensor

### Configuration
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)
# Set up
num_inputs = 3
num_parameters = num_inputs
num_observations = 100
num_tests = 100
true_sigma = torch.tensor(1.0, device=device)
true_beta = torch.ones(num_inputs, device=device)
# Output
output_date = date.today().strftime("%Y%m%d")
output_subdirectory = f"{output_date}_qposterior_test_linear_regression"


def model_func(x: Tensor, beta: Tensor) -> Tensor:
    return torch.sum(x * beta, dim=1)


def data_func(x: Tensor, beta: Tensor, gamma: float) -> Tensor:
    epsilon_mean = torch.zeros(len(x), device=device)
    epsilon_stddev = torch.sqrt(
        torch.ones(len(x), device=device) + torch.absolute(x[:, 0]) ** gamma
    )
    epsilon = torch.normal(epsilon_mean, epsilon_stddev)
    return model_func(x, beta) + true_sigma * epsilon


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, num_inputs: int) -> None:
        super().__init__()
        self.start_index_x = 0
        self.start_index_beta = num_inputs

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs[:, self.start_index_x : self.start_index_beta]
        beta = inputs[:, self.start_index_beta :]
        return model_func(x, beta)


def generate_calibration_data(
    gamma: float,
) -> tuple[tuple[CalibrationData, ...], NPArray]:
    def generate_x() -> Tensor:
        x_shape = (num_observations, num_inputs)
        x_mean = torch.zeros(x_shape, device=device)
        x_stddev = torch.ones(x_shape, device=device)
        return torch.normal(x_mean, x_stddev)

    def calculate_y(x: Tensor, beta: Tensor) -> Tensor:
        repeat_shape = (num_observations), 1
        return data_func(x, beta.repeat(repeat_shape), gamma)

    data_set_list: list[CalibrationData] = []
    true_parameters_list: list[Tensor] = []
    for _ in range(num_tests):
        x = generate_x()
        y = calculate_y(x, true_beta)
        std_noise = true_sigma.item()
        data_set = CalibrationData(inputs=x, outputs=y, std_noise=std_noise)
        data_set_list.append(data_set)
        true_parameters_list.append(true_beta)
    return (
        tuple(data_set_list),
        torch.vstack(true_parameters_list).detach().cpu().numpy(),
    )


def generate_likelihoods(
    consider_model_error: bool, calibration_data: tuple[CalibrationData, ...]
) -> tuple[Likelihood, ...]:
    if consider_model_error:
        return tuple(
            create_standard_ppinn_q_likelihood_for_noise(
                model=model,
                num_model_parameters=num_parameters,
                data=data,
                device=device,
            )
            for data in calibration_data
        )
    else:
        return tuple(
            create_standard_ppinn_likelihood_for_noise(
                model=model,
                num_model_parameters=num_parameters,
                data=data,
                device=device,
            )
            for data in calibration_data
        )


def set_up_mcmc_configs(
    likelihoods: tuple[Likelihood, ...], prior: Prior, consider_model_error: bool
) -> tuple[MetropolisHastingsConfig, ...]:
    if consider_model_error:
        std_proposal_density = torch.full(
            (num_inputs,), 0.1, dtype=torch.float64, device=device
        )
    else:
        std_proposal_density = torch.full(
            (num_inputs,), 0.1, dtype=torch.float64, device=device
        )
    configs = []
    for likelihood in likelihoods:
        cov_proposal_density = torch.diag(std_proposal_density**2)
        config = MetropolisHastingsConfig(
            likelihood=likelihood,
            prior=prior,
            initial_parameters=initial_parameters,
            num_iterations=int(1e4),
            num_burn_in_iterations=int(5e3),
            cov_proposal_density=cov_proposal_density,
        )
        configs.append(config)
    return tuple(configs)


def run_coverage_test(
    consider_model_error: bool,
    gamma: float,
    parameter_names: tuple[str, ...],
    prior: Prior,
) -> None:
    output_subdir_calibration = os.path.join(
        output_subdirectory, f"gamma_{gamma}_q_posterior_{consider_model_error}"
    )
    calibration_data, true_parameters = generate_calibration_data(gamma)
    likelihoods = generate_likelihoods(consider_model_error, calibration_data)
    mcmc_configs = set_up_mcmc_configs(likelihoods, prior, consider_model_error)
    print("############################################################")
    print(f"Q-posterior used: {consider_model_error}")
    print(f"Gamma = {gamma}")
    start = perf_counter()
    test_coverage(
        calibration_configs=mcmc_configs,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        output_subdir=output_subdir_calibration,
        project_directory=project_directory,
        device=device,
    )
    end = perf_counter()
    time = end - start
    print(f"Run time coverage test: {time}")


if __name__ == "__main__":
    lower_limits_beta = torch.full((num_inputs,), -100.0)
    upper_limits_beta = torch.full((num_inputs,), 100.0)
    prior_beta = create_multivariate_uniform_distributed_prior(
        lower_limits_beta, upper_limits_beta, device
    )
    prior = prior_beta

    parameter_names = ("beta_1", "beta_2", "beta_3")
    initial_parameters = torch.zeros(num_inputs, device=device)

    model = LinearRegressionModel(num_inputs)

    run_coverage_test(
        consider_model_error=False,
        gamma=0.0,
        parameter_names=parameter_names,
        prior=prior,
    )
    run_coverage_test(
        consider_model_error=True,
        gamma=0.0,
        parameter_names=parameter_names,
        prior=prior,
    )
    run_coverage_test(
        consider_model_error=False,
        gamma=2.0,
        parameter_names=parameter_names,
        prior=prior,
    )
    run_coverage_test(
        consider_model_error=True,
        gamma=2.0,
        parameter_names=parameter_names,
        prior=prior,
    )
