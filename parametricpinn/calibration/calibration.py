from dataclasses import dataclass
from typing import Callable, Optional, TypeAlias, cast

import numpy as np
import scipy.stats

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
from parametricpinn.types import Device, Module, MultiNormalDist, NPArray

MCMC_Algorithm: TypeAlias = MCMC_MetropolisHastings
Compiled_MCMC_Algorithm: TypeAlias = Callable[
    [], tuple[MomentsMultivariateNormal, NPArray]
]


@dataclass
class Data:
    coordinates: NPArray
    displacements: NPArray
    std_noise: float
    num_data_points: int


@dataclass
class MCMCConfig:
    parameter_names: tuple[str, ...]
    prior_means: list[float]
    prior_stds: list[float]
    initial_parameters: NPArray
    num_iterations: int


@dataclass
class MetropolisHastingsConfig(MCMCConfig):
    cov_proposal_density: NPArray


def calibrate(
    model: Module,
    data: Data,
    mcmc_config: MCMCConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    model = _load_model(
        model=model,
        output_subdir=output_subdir,
        project_directory=project_directory,
        device=device,
    )
    prior = _compile_prior(mcmc_config=mcmc_config)
    likelihood = _compile_likelihood(model=model, data=data, device=device)
    mcmc_algorithm = _compile_mcmc_algorithm(
        mcmc_config=mcmc_config,
        likelihood=likelihood,
        prior=prior,
        output_subdir=output_subdir,
        project_directory=project_directory,
    )
    return mcmc_algorithm()


def _load_model(
    model: Module,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> Module:
    print("Load model ...")
    model_loader = PytorchModelLoader(project_directory=project_directory)
    return model_loader.load(
        model=model, file_name="model_parameters", subdir_name=output_subdir
    ).to(device)


def _compile_prior(mcmc_config: MCMCConfig) -> MultiNormalDist:
    return scipy.stats.multivariate_normal(
        mean=np.array([mcmc_config.prior_means]),
        cov=np.power(np.array([mcmc_config.prior_stds]), 2),
    )


def _compile_likelihood(model: Module, data: Data, device: Device) -> LikelihoodFunc:
    covariance_error = np.diag(np.full(data.num_data_points, data.std_noise**2))
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
    prior: MultiNormalDist,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> Compiled_MCMC_Algorithm:
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
            )

        return mcmc_mh_algorithm

    else:
        raise MCMCConfigError(
            f"There is no implementation for the requested MCMC algorithm {mcmc_config}."
        )
