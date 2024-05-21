from typing import Callable, TypeAlias

import torch

from calibrationpinn.bayesian.likelihood import Likelihood
from calibrationpinn.bayesian.prior import Prior
from calibrationpinn.calibration.data import Parameters
from calibrationpinn.statistics.utility import (
    MomentsMultivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
)
from calibrationpinn.types import Device, NPArray, Tensor

MCMCOutput: TypeAlias = tuple[MomentsMultivariateNormal, NPArray]
Probability: TypeAlias = Tensor
LogUnnormalizedPosterior: TypeAlias = Callable[[Parameters], Probability]
UnnormalizedPosterior: TypeAlias = Callable[[Parameters], Probability]
Samples: TypeAlias = list[Parameters]
IsAccepted: TypeAlias = bool


def _log_unnormalized_posterior(
    likelihood: Likelihood, prior: Prior
) -> LogUnnormalizedPosterior:
    def log_unnormalized_posterior(parameters: Parameters) -> Probability:
        return likelihood.log_prob(parameters) + prior.log_prob(parameters)

    return log_unnormalized_posterior


def _unnormalized_posterior(
    likelihood: Likelihood, prior: Prior
) -> UnnormalizedPosterior:
    log_unnormalized_posterior = _log_unnormalized_posterior(likelihood, prior)

    def unnormalized_posterior(parameters: Parameters) -> Probability:
        return torch.exp(log_unnormalized_posterior(parameters))

    return unnormalized_posterior


def expand_num_iterations(num_iterations: int, num_burn_in_iterations: int) -> int:
    return num_iterations + num_burn_in_iterations


def remove_burn_in_phase(sample_list: Samples, num_burn_in_iterations: int) -> Samples:
    return sample_list[num_burn_in_iterations:]


def postprocess_samples(
    samples_list: Samples,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    samples = torch.stack(samples_list, dim=0).detach().cpu().numpy()
    posterior_moments = determine_moments_of_multivariate_normal_distribution(samples)
    return posterior_moments, samples


def evaluate_acceptance_ratio(num_accepted_proposals: int, num_iterations: int) -> None:
    acceptance_ratio = num_accepted_proposals / num_iterations
    print(f"Acceptance ratio: {round(acceptance_ratio, 4)}")


def log_bernoulli(log_probability: Tensor, device: Device) -> bool:
    """
    Runs a Bernoulli experiment on a logarithmic probability.
    Returns True with provided probability and False otherwise.

    If log_probability is nan, it will be set to 0.0.
    """
    log_probability = torch.nan_to_num(log_probability, nan=0.0)
    return bool(torch.log(torch.rand(1, device=device)) <= log_probability)
