import torch
from torch.func import vmap

from parametricpinn.bayesian.prior import Prior
from parametricpinn.calibration.bayesianinference.likelihoods.likelihoodstrategies import (
    LikelihoodStrategy,
)
from parametricpinn.types import Device, Tensor


class LogMarginalLikelihood(torch.nn.Module):
    def __init__(
        self,
        likelihood: LikelihoodStrategy,
        initial_hyperparameters: Tensor,
        num_material_parameter_samples: int,
        prior_material_parameters: Prior,
        device: Device,
    ) -> None:
        super().__init__()
        self._likelihood = likelihood
        self._hyperparameters = torch.nn.Parameter(
            initial_hyperparameters.type(torch.float64).to(device),
            requires_grad=True,
        )
        self._num_material_parameter_samples = num_material_parameter_samples
        self._material_parameter_samples = (
            prior_material_parameters.sample((num_material_parameter_samples, 1))
            .detach()
            .to(device)
        )
        self._log_probs_prior_material_parameters = (
            vmap(prior_material_parameters.log_prob)(self._material_parameter_samples)
            .detach()
            .to(device)
        )
        self._device = device

    def forward(self) -> Tensor:
        return self._log_prob()

    def get_hyperparameters(self) -> Tensor:
        return self._hyperparameters.data.detach()

    def _log_prob(self) -> Tensor:
        parameters = torch.concat(
            (
                self._material_parameter_samples,
                self._hyperparameters.repeat((self._num_material_parameter_samples, 1)),
            ),
            dim=1,
        ).to(self._device)
        # log_probs_likelihood = vmap(self._likelihood.log_prob)(parameters)
        log_probs_likelihood = torch.concat(
            [
                torch.unsqueeze(self._likelihood.log_prob(parameter), dim=0)
                for parameter in parameters
            ]
        )
        log_probs = log_probs_likelihood + self._log_probs_prior_material_parameters
        return self._logarithmic_sum_of_exponentials(log_probs)

    def _logarithmic_sum_of_exponentials(self, log_probs: Tensor) -> Tensor:
        max_log_prob = torch.max(log_probs)
        return max_log_prob + torch.log(torch.sum(torch.exp(log_probs - max_log_prob)))


def optimize_hyperparameters(
    likelihood: LikelihoodStrategy,
    initial_hyperparameters: Tensor,
    prior_material_parameters: Prior,
    num_material_parameter_samples: int,
    num_iterations: int,
    device: Device,
) -> Tensor:
    log_marginal_likelihood = LogMarginalLikelihood(
        likelihood=likelihood,
        initial_hyperparameters=initial_hyperparameters.clone(),
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
        print(f"Hyperparameters: {log_marginal_likelihood.get_hyperparameters()}")
        optimizer.step(loss_func_closure)

    return log_marginal_likelihood.get_hyperparameters().detach()
