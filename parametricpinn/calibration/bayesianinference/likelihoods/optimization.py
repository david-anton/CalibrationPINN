from typing import TypeAlias

import numpy as np
import pandas as pd
import torch

from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.bayesianinference.likelihoods.likelihoodstrategies import (
    OptimizedLikelihoodStrategy,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.types import Device, Tensor


class LogMarginalLikelihood(torch.nn.Module):
    def __init__(
        self,
        likelihood: OptimizedLikelihoodStrategy,
        num_material_parameter_samples: int,
        prior_material_parameters: Prior,
        device: Device,
    ) -> None:
        super().__init__()
        self._likelihood = likelihood
        self._material_parameter_samples = (
            prior_material_parameters.sample((num_material_parameter_samples, 1))
            .detach()
            .requires_grad_(False)
            .to(device)
        )
        self._log_probs_prior_material_parameters = (
            torch.concat(
                [
                    torch.unsqueeze(prior_material_parameters.log_prob(sample), dim=0)
                    for sample in self._material_parameter_samples
                ],
            )
            .detach()
            .requires_grad_(False)
            .to(device)
        )
        self._device = device

    def forward(self) -> Tensor:
        return self._log_prob()

    def _log_prob(self) -> Tensor:
        log_probs_likelihood = torch.concat(
            [
                torch.unsqueeze(self._likelihood.log_prob(material_parameter), dim=0)
                for material_parameter in self._material_parameter_samples
            ]
        )
        log_probs = log_probs_likelihood + self._log_probs_prior_material_parameters
        return self._logarithmic_sum_of_exponentials(log_probs)

    def _logarithmic_sum_of_exponentials(self, log_probs: Tensor) -> Tensor:
        max_log_prob = torch.max(log_probs)
        return max_log_prob + torch.log(torch.sum(torch.exp(log_probs - max_log_prob)))


def optimize_likelihood_hyperparameters(
    likelihood: OptimizedLikelihoodStrategy,
    prior_material_parameters: Prior,
    num_material_parameter_samples: int,
    num_iterations: int,
    device: Device,
) -> None:
    print("Start optimization of likelihood parameters ...")
    log_marginal_likelihood = LogMarginalLikelihood(
        likelihood=likelihood,
        num_material_parameter_samples=num_material_parameter_samples,
        prior_material_parameters=prior_material_parameters,
        device=device,
    )

    def loss_func() -> Tensor:
        return -log_marginal_likelihood()

    optimizer = torch.optim.LBFGS(
        params=log_marginal_likelihood.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def loss_func_closure() -> float:
        optimizer.zero_grad()
        loss = loss_func()
        loss.backward()
        return loss.item()

    for _ in range(num_iterations):
        optimizer.step(loss_func_closure)


def save_optimized_likelihood_hyperparameters(
    likelihood: OptimizedLikelihoodStrategy,
    file_name_prefix: str,
    test_case_index: int,
    output_subdirectory: str,
    project_directory: ProjectDirectory,
) -> None:
    named_parameters = likelihood.get_named_parameters()
    header = tuple(key for key, _ in named_parameters.items())
    parameters = np.array(
        tuple(value.item() for _, value in named_parameters.items())
    ).reshape((1, -1))
    data_frame = pd.DataFrame(parameters, columns=header)
    data_writer = PandasDataWriter(project_directory)
    file_name = f"{file_name_prefix}_{test_case_index}.csv"
    data_writer.write(
        data=data_frame,
        file_name=file_name,
        subdir_name=output_subdirectory,
        header=header,
    )
