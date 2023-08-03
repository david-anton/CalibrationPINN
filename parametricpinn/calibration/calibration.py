from typing import Callable, TypeAlias, Union, cast

import torch

from parametricpinn.calibration.bayesian.likelihood import (
    LikelihoodFunc,
    compile_likelihood,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.mcmc_hamiltonian import (
    HamiltonianConfig,
    MCMCHamiltonianFunc,
    mcmc_hamiltonian,
)
from parametricpinn.calibration.bayesian.mcmc_metropolishastings import (
    MCMCMetropolisHastingsFunc,
    MetropolisHastingsConfig,
    mcmc_metropolishastings,
)
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.calibration.data import CalibrationData, PreprocessedData
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.errors import MCMCConfigError, UnvalidCalibrationDataError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.loaderssavers import PytorchModelLoader
from parametricpinn.types import Device, Module, NPArray, Tensor, TorchMultiNormalDist

MCMC_Algorithm: TypeAlias = Union[MCMCMetropolisHastingsFunc, MCMCHamiltonianFunc]
MCMC_Algorithm_Closure: TypeAlias = Callable[
    [], tuple[MomentsMultivariateNormal, NPArray]
]


def calibrate(
    model: Module,
    calibration_data: CalibrationData,
    mcmc_config: MCMCConfig,
    name_model_parameters_file: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> tuple[MomentsMultivariateNormal, NPArray]:
    preprocessed_data = _preprocess_calibration_data(calibration_data)
    model = _load_model(
        model=model,
        name_model_parameters_file=name_model_parameters_file,
        output_subdir=output_subdir,
        project_directory=project_directory,
        device=device,
    )
    prior = _compile_prior(mcmc_config=mcmc_config, device=device)
    likelihood = _compile_likelihood(model=model, data=preprocessed_data, device=device)
    mcmc_algorithm = _compile_mcmc_algorithm(
        mcmc_config=mcmc_config,
        likelihood=likelihood,
        prior=prior,
        output_subdir=output_subdir,
        project_directory=project_directory,
        device=device,
    )
    return mcmc_algorithm()


def _preprocess_calibration_data(data: CalibrationData) -> PreprocessedData:
    _validate_calibration_data(data)
    outputs = data.outputs
    num_data_points = outputs.size()[0]
    dim_outputs = outputs.size()[1]

    return PreprocessedData(
        inputs=data.inputs,
        outputs=data.outputs,
        std_noise=data.std_noise,
        num_data_points=num_data_points,
        dim_outputs=dim_outputs,
    )


def _validate_calibration_data(calibration_data: CalibrationData) -> None:
    inputs = calibration_data.inputs
    outputs = calibration_data.outputs
    if not inputs.size()[0] == outputs.size()[0]:
        raise UnvalidCalibrationDataError(
            "Size of input and output data does not match."
        )


def _load_model(
    model: Module,
    name_model_parameters_file: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> Module:
    model_loader = PytorchModelLoader(project_directory=project_directory)
    model = model_loader.load(
        model=model, file_name=name_model_parameters_file, subdir_name=output_subdir
    ).to(device)
    freeze_model(model)
    return model


def _compile_prior(mcmc_config: MCMCConfig, device: Device) -> TorchMultiNormalDist:
    def _set_up_covariance_matrix(prior_stds: list[float]) -> Tensor:
        if len(prior_stds) == 1:
            return torch.unsqueeze(
                torch.tensor(prior_stds, dtype=torch.float64, device=device) ** 2, dim=1
            )
        else:
            return torch.diag(
                torch.tensor(prior_stds, dtype=torch.float64, device=device) ** 2
            )

    return torch.distributions.MultivariateNormal(
        loc=torch.tensor(mcmc_config.prior_means, device=device),
        covariance_matrix=_set_up_covariance_matrix(mcmc_config.prior_stds),
    )


def _compile_likelihood(
    model: Module, data: PreprocessedData, device: Device
) -> LikelihoodFunc:
    return compile_likelihood(
        model=model,
        data=data,
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
        config_mh = cast(MetropolisHastingsConfig, mcmc_config)

        def mcmc_mh_algorithm() -> tuple[MomentsMultivariateNormal, NPArray]:
            return mcmc_metropolishastings(
                parameter_names=config_mh.parameter_names,
                true_parameters=config_mh.true_parameters,
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_mh.initial_parameters,
                cov_proposal_density=config_mh.cov_proposal_density,
                num_iterations=config_mh.num_iterations,
                num_burn_in_iterations=config_mh.num_burn_in_iterations,
                output_subdir=output_subdir,
                project_directory=project_directory,
                device=device,
            )

        return mcmc_mh_algorithm

    elif isinstance(mcmc_config, HamiltonianConfig):
        config_h = cast(HamiltonianConfig, mcmc_config)

        def mcmc_hamiltonian_algorithm() -> tuple[MomentsMultivariateNormal, NPArray]:
            return mcmc_hamiltonian(
                parameter_names=config_h.parameter_names,
                true_parameters=config_h.true_parameters,
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_h.initial_parameters,
                num_leapfrog_steps=config_h.num_leabfrog_steps,
                leapfrog_step_size=config_h.leapfrog_step_size,
                num_iterations=config_h.num_iterations,
                num_burn_in_iterations=config_h.num_burn_in_iterations,
                output_subdir=output_subdir,
                project_directory=project_directory,
                device=device,
            )

        return mcmc_hamiltonian_algorithm
    else:
        raise MCMCConfigError(
            f"There is no implementation for the requested MCMC algorithm {mcmc_config}."
        )
