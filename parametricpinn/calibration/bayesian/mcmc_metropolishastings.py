from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from parametricpinn.calibration.bayesian.likelihood import LikelihoodFunc
from parametricpinn.calibration.bayesian.mcmc_base import (
    Samples,
    compile_unnnormalized_posterior,
    evaluate_acceptance_ratio,
    expand_num_iterations,
    postprocess_samples,
    remove_burn_in_phase,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.prior import PriorFunc
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, NPArray, Tensor, TorchMultiNormalDist

MCMCMetropolisHastingsFunc: TypeAlias = Callable[
    [
        tuple[str, ...],
        tuple[float, ...],
        LikelihoodFunc,
        PriorFunc,
        Tensor,
        Tensor,
        int,
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
    prior: PriorFunc,
    initial_parameters: Tensor,
    cov_proposal_density: Tensor,
    num_iterations: int,
    num_burn_in_iterations: int,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    num_total_iterations = expand_num_iterations(
        num_iterations=num_iterations, num_burn_in_iterations=num_burn_in_iterations
    )
    unnormalized_posterior = compile_unnnormalized_posterior(
        likelihood=likelihood, prior=prior
    )

    def compile_proposal_density(
        initial_parameters: Tensor, cov_proposal_density: Tensor
    ) -> TorchMultiNormalDist:
        if cov_proposal_density.size() == (1,):
            cov_proposal_density = torch.unsqueeze(cov_proposal_density, dim=1)
        cov_proposal_density = cov_proposal_density.type(torch.float64)

        return torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(
                initial_parameters, dtype=torch.float64, device=device
            ),
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

    def metropolis_hastings_update(state: MHUpdateState) -> tuple[Tensor, bool]:
        acceptance_ratio = torch.minimum(
            torch.tensor(1.0, device=device),
            unnormalized_posterior(state.next_parameters)
            / unnormalized_posterior(state.parameters),
        )
        rand_uniform_number = torch.squeeze(torch.rand(1, device=device), 0)
        next_parameters = state.next_parameters
        is_accepted = True
        if rand_uniform_number > acceptance_ratio:
            next_parameters = state.parameters
            is_accepted = False
        return next_parameters, is_accepted

    def one_iteration(parameters: Tensor) -> tuple[Tensor, bool]:
        next_parameters = propose_next_parameters(parameters)
        mh_update_state = MHUpdateState(
            parameters=parameters,
            next_parameters=next_parameters,
        )
        return metropolis_hastings_update(mh_update_state)

    samples_list: Samples = []
    num_accepted_proposals = 0
    parameters = initial_parameters
    for i in range(num_total_iterations):
        parameters, is_accepted = one_iteration(parameters)
        samples_list.append(parameters)
        if i < num_burn_in_iterations and is_accepted:
            num_accepted_proposals += 1

    samples_list = remove_burn_in_phase(
        sample_list=samples_list, num_burn_in_iterations=num_burn_in_iterations
    )
    moments, samples = postprocess_samples(
        samples_list=samples_list,
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        mcmc_algorithm="metropolishastings_mcmc",
        output_subdir=output_subdir,
        project_directory=project_directory,
    )
    evaluate_acceptance_ratio(
        num_accepted_proposals=num_accepted_proposals, num_iterations=num_iterations
    )
    return moments, samples
