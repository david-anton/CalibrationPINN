from typing import TypeAlias

import numpy as np
import scipy.stats

from parametricpinn.calibration.likelihood import LikelihoodFunc
from parametricpinn.calibration.plot import plot_posterior_normal_distributions
from parametricpinn.calibration.statistics import (
    _determine_moments_of_multivariate_normal_distribution,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import MultiNormalDist, NPArray

Samples: TypeAlias = list[NPArray]


def mcmc_metropolishastings(
    parameter_names: tuple[str, ...],
    likelihood: LikelihoodFunc,
    prior: MultiNormalDist,
    initial_parameters: NPArray,
    std_proposal_density: NPArray,
    num_iterations: int,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> tuple[MultiNormalDist, NPArray]:
    unnormalized_posterior = lambda parameters: likelihood(parameters) * prior.pdf(
        parameters
    )
    proposal_density = scipy.stats.multivariate_normal(
        np.zeros_like(initial_parameters), std_proposal_density
    )

    samples_list: Samples = []

    def one_iteration(parameters: NPArray) -> NPArray:
        next_parameters = parameters + proposal_density.rvs()
        acceptance_ratio = unnormalized_posterior(next_parameters) / unnormalized_posterior(
            parameters
        )
        rand_uniform_number = scipy.stats.uniform.rvs(size=1)
        if rand_uniform_number > acceptance_ratio:
            next_parameters = parameters
        samples_list.append(next_parameters)
        return next_parameters

    parameters = initial_parameters
    for _ in range(num_iterations):
        parameters = one_iteration(parameters)

    samples = np.array(samples_list)
    posterior_moments = _determine_moments_of_multivariate_normal_distribution(samples)
    plot_posterior_normal_distributions(
        parameter_names, posterior_moments, samples, output_subdir, project_directory
    )
    return posterior_moments, samples
