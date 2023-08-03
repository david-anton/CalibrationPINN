from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.calibration.bayesian.mcmc_base import (
    Samples,
    compile_unnnormalized_posterior,
    postprocess_samples,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, NPArray, Tensor, TorchMultiNormalDist

MCMCMetropolisHastingsFunc: TypeAlias = Callable[
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
        Device,
    ],
    tuple[MomentsMultivariateNormal, NPArray],
]


@dataclass
class MetropolisHastingsConfig(MCMCConfig):
    cov_proposal_density: Tensor


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
    print("MCMC algorithm used: Metropolis Hastings")
    unnormalized_posterior = compile_unnnormalized_posterior(likelihood, prior)

    def compile_proposal_density(
        initial_parameters: Tensor, cov_proposal_density: Tensor
    ) -> TorchMultiNormalDist:
        if cov_proposal_density.size() == (1,):
            cov_proposal_density = torch.unsqueeze(cov_proposal_density, dim=1)
        cov_proposal_density = cov_proposal_density.to(torch.float)

        return torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(initial_parameters, dtype=torch.float, device=device),
            covariance_matrix=cov_proposal_density,
        )

    proposal_density = compile_proposal_density(
        initial_parameters, cov_proposal_density
    )

    def propose_next_parameters(parameters: Tensor) -> Tensor:
        return parameters + proposal_density.sample()

    @dataclass
    class MHUpdateState:
        parameters: Tensor
        next_parameters: Tensor

    def metropolis_hastings_update(state: MHUpdateState) -> Tensor:
        acceptance_ratio = torch.minimum(
            torch.tensor(1.0, device=device),
            unnormalized_posterior(state.next_parameters)
            / unnormalized_posterior(state.parameters),
        )
        rand_uniform_number = torch.squeeze(torch.rand(1, device=device), 0)
        next_parameters = state.next_parameters
        if rand_uniform_number > acceptance_ratio:
            next_parameters = state.parameters
        return next_parameters

    def one_iteration(parameters: Tensor) -> Tensor:
        next_parameters = propose_next_parameters(parameters)
        mh_update_state = MHUpdateState(
            parameters=parameters,
            next_parameters=next_parameters,
        )
        return metropolis_hastings_update(mh_update_state)

    samples_list: Samples = []
    parameters = initial_parameters
    for _ in range(num_iterations):
        parameters = one_iteration(parameters)
        samples_list.append(parameters)

    moments, samples = postprocess_samples(
        samples_list=samples_list,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        output_subdir=output_subdir,
        project_directory=project_directory,
    )
    return moments, samples
