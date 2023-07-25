from dataclasses import dataclass
from typing import Callable, TypeAlias, cast

import torch

from parametricpinn.calibration.bayesian.likelihood import (
    LikelihoodFunc,
    compile_likelihood,
)
from parametricpinn.calibration.bayesian.mcmc_metropolishastings import (
    MCMC_MetropolisHastings,
    mcmc_metropolishastings,
)
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.errors import MCMCConfigError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.loaderssavers import PytorchModelLoader
from parametricpinn.types import Device, Module, NPArray, Tensor, TorchMultiNormalDist

MCMC_Algorithm: TypeAlias = MCMC_MetropolisHastings
MCMC_Algorithm_Closure: TypeAlias = Callable[
    [], tuple[MomentsMultivariateNormal, NPArray]
]


@dataclass
class Data:
    coordinates: Tensor
    displacements: Tensor
    std_noise: float
    num_data_points: int


@dataclass
class MCMCConfig:
    parameter_names: tuple[str, ...]
    prior_means: list[float]
    prior_stds: list[float]
    initial_parameters: Tensor
    num_iterations: int


@dataclass
class MetropolisHastingsConfig(MCMCConfig):
    cov_proposal_density: Tensor


def calibrate(
    model: Module,
    data: Data,
    mcmc_config: MCMCConfig,
    name_model_parameters_file: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    model = _load_model(
        model=model,
        name_model_parameters_file=name_model_parameters_file,
        output_subdir=output_subdir,
        project_directory=project_directory,
        device=device,
    )
    prior = _compile_prior(mcmc_config=mcmc_config, device=device)
    likelihood = _compile_likelihood(model=model, data=data, device=device)
    mcmc_algorithm = _compile_mcmc_algorithm(
        mcmc_config=mcmc_config,
        likelihood=likelihood,
        prior=prior,
        output_subdir=output_subdir,
        project_directory=project_directory,
        device=device,
    )
    return mcmc_algorithm()


def _load_model(
    model: Module,
    name_model_parameters_file: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> Module:
    print("Load model ...")
    model_loader = PytorchModelLoader(project_directory=project_directory)
    return model_loader.load(
        model=model, file_name=name_model_parameters_file, subdir_name=output_subdir
    ).to(device)


def _compile_prior(mcmc_config: MCMCConfig, device: Device) -> TorchMultiNormalDist:
    def _set_up_covariance_matrix(prior_stds: list[float]) -> Tensor:
        if len(prior_stds) == 1:
            return torch.unsqueeze(
                torch.tensor(prior_stds, dtype=torch.float, device=device) ** 2, dim=1
            )
        else:
            return torch.diag(
                torch.tensor(prior_stds, dtype=torch.float, device=device) ** 2
            )

    return torch.distributions.MultivariateNormal(
        loc=torch.tensor(mcmc_config.prior_means, device=device),
        covariance_matrix=_set_up_covariance_matrix(mcmc_config.prior_stds),
    )


def _compile_likelihood(model: Module, data: Data, device: Device) -> LikelihoodFunc:
    covariance_error = torch.diag(
        torch.full((data.num_data_points,), data.std_noise**2)
    )
    return compile_likelihood(
        model=model,
        coordinates=data.coordinates,
        data=data.displacements,
        covariance_error=covariance_error,
        device=device,
    )


def _compile_mcmc_algorithm(
    mcmc_config: MCMCConfig,
    likelihood: LikelihoodFunc,
    prior: TorchMultiNormalDist,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> MCMC_Algorithm_Closure:
    if isinstance(mcmc_config, MetropolisHastingsConfig):
        mcmc_mh_config = cast(MetropolisHastingsConfig, mcmc_config)

        def mcmc_mh_algorithm() -> tuple[MomentsMultivariateNormal, NPArray]:
            return mcmc_metropolishastings(
                parameter_names=mcmc_mh_config.parameter_names,
                likelihood=likelihood,
                prior=prior,
                initial_parameters=mcmc_mh_config.initial_parameters,
                cov_proposal_density=mcmc_mh_config.cov_proposal_density,
                num_iterations=mcmc_mh_config.num_iterations,
                output_subdir=output_subdir,
                project_directory=project_directory,
                device=device,
            )

        return mcmc_mh_algorithm

    else:
        raise MCMCConfigError(
            f"There is no implementation for the requested MCMC algorithm {mcmc_config}."
        )
