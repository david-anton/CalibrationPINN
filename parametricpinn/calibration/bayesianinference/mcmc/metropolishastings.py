from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch

from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.bayesianinference.mcmc.base import (
    IsAccepted,
    MCMCOutput,
    Parameters,
    Probability,
    Samples,
    _log_unnormalized_posterior,
    evaluate_acceptance_ratio,
    expand_num_iterations,
    log_bernoulli,
    postprocess_samples,
    remove_burn_in_phase,
)
from parametricpinn.calibration.bayesianinference.mcmc.config import MCMCConfig
from parametricpinn.types import Device, Tensor, TorchMultiNormalDist

CovarianceProposalDensity: TypeAlias = Tensor

MCMCMetropolisHastingsFunc: TypeAlias = Callable[
    [
        Likelihood,
        Prior,
        Parameters,
        CovarianceProposalDensity,
        int,
        int,
        Device,
    ],
    MCMCOutput,
]


@dataclass
class MetropolisHastingsConfig(MCMCConfig):
    cov_proposal_density: Tensor


def mcmc_metropolishastings(
    likelihood: Likelihood,
    prior: Prior,
    initial_parameters: Parameters,
    cov_proposal_density: CovarianceProposalDensity,
    num_iterations: int,
    num_burn_in_iterations: int,
    device: Device,
) -> MCMCOutput:
    num_total_iterations = expand_num_iterations(num_iterations, num_burn_in_iterations)
    log_unnorm_posterior = _log_unnormalized_posterior(likelihood, prior)

    def compile_proposal_density(
        initial_parameters: Parameters, cov_proposal_density: CovarianceProposalDensity
    ) -> TorchMultiNormalDist:
        if cov_proposal_density.size() == (1,):
            cov_proposal_density = torch.unsqueeze(cov_proposal_density, dim=1)
        cov_proposal_density = cov_proposal_density.type(torch.float64)

        return torch.distributions.MultivariateNormal(
            loc=torch.zeros_like(
                initial_parameters, dtype=torch.float64, device=device
            ),
            covariance_matrix=cov_proposal_density.to(device),
        )

    proposal_density = compile_proposal_density(
        initial_parameters, cov_proposal_density
    )

    @dataclass
    class CurrentParameters:
        parameters: Parameters
        log_unnorm_posterior_prob: Probability
        is_accepted: IsAccepted

    def metropolis_hastings_sampler(
        current_parameters: CurrentParameters,
    ) -> CurrentParameters:
        @dataclass
        class MHUpdateInput:
            parameters: Parameters
            log_unnorm_posterior_prob: Probability
            next_parameters: Parameters

        def propose_next_parameters(parameters: Parameters) -> Parameters:
            return parameters + proposal_density.sample()

        def metropolis_hastings_update(upate_input: MHUpdateInput) -> CurrentParameters:
            log_unnorm_posterior_prob = upate_input.log_unnorm_posterior_prob
            next_parameters = upate_input.next_parameters

            next_log_unnorm_posterior_prob = log_unnorm_posterior(next_parameters)
            log_acceptance_prob = (
                next_log_unnorm_posterior_prob - log_unnorm_posterior_prob
            )

            is_accepted = True
            if not log_bernoulli(log_acceptance_prob, device):
                next_parameters = upate_input.parameters
                next_log_unnorm_posterior_prob = log_unnorm_posterior_prob
                is_accepted = False

            return CurrentParameters(
                parameters=next_parameters,
                log_unnorm_posterior_prob=next_log_unnorm_posterior_prob,
                is_accepted=is_accepted,
            )

        parameters = current_parameters.parameters
        log_unnorm_posterior_prob = current_parameters.log_unnorm_posterior_prob
        next_parameters = propose_next_parameters(parameters)
        update_input = MHUpdateInput(
            parameters=parameters,
            log_unnorm_posterior_prob=log_unnorm_posterior_prob,
            next_parameters=next_parameters,
        )
        return metropolis_hastings_update(update_input)

    def one_iteration(current_parameters: CurrentParameters) -> CurrentParameters:
        return metropolis_hastings_sampler(current_parameters)

    samples_list: Samples = []
    num_accepted_proposals = 0
    parameters = initial_parameters.to(device)
    current_parameters = CurrentParameters(
        parameters=parameters,
        log_unnorm_posterior_prob=log_unnorm_posterior(parameters),
        is_accepted=False,
    )
    for i in range(num_total_iterations):
        current_parameters = one_iteration(current_parameters)
        samples_list.append(current_parameters.parameters)
        if i > num_burn_in_iterations and current_parameters.is_accepted:
            num_accepted_proposals += 1

    samples_list = remove_burn_in_phase(
        sample_list=samples_list, num_burn_in_iterations=num_burn_in_iterations
    )
    moments, samples = postprocess_samples(samples_list=samples_list)
    evaluate_acceptance_ratio(
        num_accepted_proposals=num_accepted_proposals, num_iterations=num_iterations
    )
    return moments, samples
