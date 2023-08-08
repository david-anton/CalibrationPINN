from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.calibration.bayesian.plot import plot_posterior_normal_distributions
from parametricpinn.calibration.bayesian.prior import PriorFunc
from parametricpinn.calibration.bayesian.statistics import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import NPArray, Tensor, TorchMultiNormalDist

UnnormalizedPosterior: TypeAlias = Callable[[Tensor], Tensor]
Samples: TypeAlias = list[Tensor]


def compile_unnnormalized_posterior(
    likelihood: LikelihoodFunc, prior: PriorFunc
) -> UnnormalizedPosterior:
    def _unnormalized_posterior(parameters: Tensor) -> Tensor:
        return likelihood(parameters) * prior(parameters)

    return _unnormalized_posterior


def correct_num_iterations(num_iterations: int, num_burn_in_iterations: int) -> int:
    return num_iterations + num_burn_in_iterations


def remove_burn_in_phase(sample_list: Samples, num_burn_in_iterations: int) -> Samples:
    return sample_list[num_burn_in_iterations:]


def postprocess_samples(
    samples_list: Samples,
    parameter_names: tuple[str, ...],
    true_parameters: tuple[float, ...],
    mcmc_algorithm: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    samples = torch.stack(samples_list, dim=0).detach().cpu().numpy()
    posterior_moments = determine_moments_of_multivariate_normal_distribution(samples)
    plot_posterior_normal_distributions(
        parameter_names,
        true_parameters,
        posterior_moments,
        samples,
        mcmc_algorithm,
        output_subdir,
        project_directory,
    )
    return posterior_moments, samples


def evaluate_acceptance_ratio(num_accepted_proposals: int, num_iterations: int) -> None:
    acceptance_ratio = num_accepted_proposals / num_iterations
    print(f"Acceptance ratio: {round(acceptance_ratio, 3)}")
