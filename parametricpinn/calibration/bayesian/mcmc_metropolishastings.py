from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.calibration.bayesian.plot import plot_posterior_normal_distributions
from parametricpinn.calibration.bayesian.statistics import (
    MomentsMultivariateNormal,
    _determine_moments_of_multivariate_normal_distribution,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, NPArray, Tensor, TorchMultiNormalDist

Samples: TypeAlias = list[Tensor]
MCMC_MetropolisHastings: TypeAlias = Callable[
    [
        tuple[str, ...],
        tuple[float, ...],
        LikelihoodFunc,
        TorchMultiNormalDist,
        Tensor,
        Tensor,
        int,
        str,
        ProjectDirectory,
    ],
    tuple[MomentsMultivariateNormal, NPArray],
]
UnnormalizedPosterior: TypeAlias = Callable[[Tensor], Tensor]


def mcmc_metropolishastings(
    parameter_names: tuple[str, ...],
    true_parameters: tuple[float, ...],
    likelihood: LikelihoodFunc,
    prior: TorchMultiNormalDist,
    initial_parameters: Tensor,
    cov_proposal_density: Tensor,
    num_iterations: int,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    def _compile_unnnormalized_posterior(
        likelihood: LikelihoodFunc, prior: TorchMultiNormalDist
    ) -> UnnormalizedPosterior:
        def _unnormalized_posterior(parameters: Tensor) -> Tensor:
            return likelihood(parameters) * torch.pow(10, prior.log_prob(parameters))

        return _unnormalized_posterior

    def _compile_proposal_density(
        initial_parameters: Tensor, cov_proposal_density: Tensor
    ) -> TorchMultiNormalDist:
        if cov_proposal_density.size() == (1,):
            cov_proposal_density = torch.unsqueeze(cov_proposal_density, dim=1)
        cov_proposal_density = cov_proposal_density.to(torch.float)

        return torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(initial_parameters, dtype=torch.float, device=device),
            covariance_matrix=cov_proposal_density,
        )

    unnormalized_posterior = _compile_unnnormalized_posterior(likelihood, prior)
    proposal_density = _compile_proposal_density(
        initial_parameters, cov_proposal_density
    )
    samples_list: Samples = []

    def one_iteration(parameters: Tensor) -> Tensor:
        next_parameters = parameters + proposal_density.sample()
        acceptance_ratio = torch.minimum(
            torch.tensor(1.0, device=device),
            unnormalized_posterior(next_parameters)
            / unnormalized_posterior(parameters),
        )
        rand_uniform_number = torch.squeeze(torch.rand(1, device=device), 0)
        if rand_uniform_number > acceptance_ratio:
            next_parameters = parameters
        samples_list.append(next_parameters)
        return next_parameters

    parameters = initial_parameters
    for _ in range(num_iterations):
        parameters = one_iteration(parameters)

    samples = torch.stack(samples_list, dim=0).detach().cpu().numpy()
    posterior_moments = _determine_moments_of_multivariate_normal_distribution(samples)
    plot_posterior_normal_distributions(
        parameter_names,
        true_parameters,
        posterior_moments,
        samples,
        output_subdir,
        project_directory,
    )
    return posterior_moments, samples
