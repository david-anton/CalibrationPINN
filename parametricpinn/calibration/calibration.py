from typing import Callable, TypeAlias, cast

import torch

from parametricpinn.calibration.bayesian.likelihood import (
    LikelihoodFunc,
    compile_likelihood,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.mcmc_metropolishastings import (
    MCMC_MetropolisHastings_func,
    MetropolisHastingsConfig,
    mcmc_metropolishastings,
)
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.calibration.data import CalibrationData, PreprocessedData
from parametricpinn.calibration.utility import _freeze_model
from parametricpinn.errors import MCMCConfigError, UnvalidCalibrationDataError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.loaderssavers import PytorchModelLoader
from parametricpinn.types import Device, Module, NPArray, Tensor, TorchMultiNormalDist

MCMC_Algorithm: TypeAlias = MCMC_MetropolisHastings_func
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
    error_covariance_matrix = _create_error_covariance_matrix(
        data=data, num_data_points=num_data_points, dim_outputs=dim_outputs
    )
    return PreprocessedData(
        inputs=data.inputs,
        outputs=data.outputs,
        std_noise=data.std_noise,
        error_covariance_matrix=error_covariance_matrix,
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


def _create_error_covariance_matrix(
    data: CalibrationData, num_data_points: int, dim_outputs: int
) -> Tensor:
    return torch.diag(torch.full((num_data_points * dim_outputs,), data.std_noise**2))


def _load_model(
    model: Module,
    name_model_parameters_file: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> Module:
    print("Load model ...")
    model_loader = PytorchModelLoader(project_directory=project_directory)
    model = model_loader.load(
        model=model, file_name=name_model_parameters_file, subdir_name=output_subdir
    ).to(device)
    _freeze_model(model)
    return model


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
        mcmc_mh_config = cast(MetropolisHastingsConfig, mcmc_config)

        def mcmc_mh_algorithm() -> tuple[MomentsMultivariateNormal, NPArray]:
            return mcmc_metropolishastings(
                parameter_names=mcmc_mh_config.parameter_names,
                true_parameters=mcmc_mh_config.true_parameters,
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
