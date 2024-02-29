import os
from datetime import date

import numpy as np
import pandas as pd
import torch
from torch.func import vmap

from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import create_univariate_normal_distributed_prior
from parametricpinn.calibration import (
    CalibrationData,
    MetropolisHastingsConfig,
    test_coverage,
    test_least_squares_calibration,
)
from parametricpinn.calibration.bayesianinference.likelihoods import (
    create_standard_ppinn_likelihood_for_noise,
    create_standard_ppinn_q_likelihood_for_noise,
)
from parametricpinn.data.parameterssampling import sample_uniform_grid
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader, PandasDataWriter
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.types import Device, NPArray, Tensor

from parametricpinn.bayesian.prior import create_multivariate_uniform_distributed_prior


### Configuration
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)
# Set up
num_parameters = 1
num_inputs = 3
num_observations = 100
num_tests = 0
gamma = 0.0
true_sigma = torch.tensor(1.0, device=device)
true_beta = torch.ones(num_inputs, device=device)
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_qposterior_test"


def model(x: Tensor, beta: Tensor, Tensor, gamma: float) -> Tensor:
    def vmap_model(x: Tensor, beta: Tensor) -> Tensor:
        epsilon_mean = torch.tensor(0.0, device=device)
        epsilon_stddev = torch.sqrt(torch.tensor(1.0, device=device) + x[0] ** gamma)
        epsilon = torch.normal(epsilon_mean, epsilon_stddev)
        return torch.matmul(x, beta) + true_sigma * epsilon

    return vmap(vmap_model)(x, beta)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, num_inputs: int) -> None:
        super().__init__()
        self.gamma = 0
        self.start_index_x = 0
        self.start_index_beta = num_inputs

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs[self.start_index_x, self.start_index_beta]
        beta = inputs[self.start_index_beta, self.start_index_sigma]
        return model(x, beta, self.gamma)


def generate_data(gamma: float) -> tuple[CalibrationData, ...]:
    def generate_x() -> Tensor:
        x_shape = (num_observations, num_inputs)
        x_mean = torch.zeros(x_shape, device=device)
        x_stddev = torch.ones(x_shape, device=device)
        return torch.normal(x_mean, x_stddev)

    def calculate_y(x: Tensor) -> Tensor:
        repeat_shape = (num_observations), 1
        return model(x, true_beta.repeat(repeat_shape), gamma)

    data_sets: list[CalibrationData] = []
    for _ in range(num_tests):
        x = generate_x()
        y = calculate_y(x)
        std_noise = true_sigma.item()
        data_set = CalibrationData(inputs=x, outputs=y, std_noise=std_noise)
        data_sets.append(data_set)
    return tuple(data_sets)


lower_limits_beta = torch.full(num_inputs, -100.0)
upper_limits_beta = torch.full(num_inputs, 100.0)
prior_beta = create_multivariate_uniform_distributed_prior(
    lower_limits_beta, upper_limits_beta
)
