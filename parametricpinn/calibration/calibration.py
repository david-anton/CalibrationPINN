from typing import Callable, TypeAlias, cast

from parametricpinn.calibration.bayesian.likelihood import (
    LikelihoodFunc,
    compile_likelihood,
)
from parametricpinn.calibration.bayesian.mcmc_config import MCMCConfig
from parametricpinn.calibration.bayesian.mcmc_efficientnuts import (
    EfficientNUTSConfig,
    mcmc_efficientnuts,
)
from parametricpinn.calibration.bayesian.mcmc_hamiltonian import (
    HamiltonianConfig,
    mcmc_hamiltonian,
)
from parametricpinn.calibration.bayesian.mcmc_metropolishastings import (
    MetropolisHastingsConfig,
    mcmc_metropolishastings,
)
from parametricpinn.calibration.bayesian.mcmc_naivenuts import (
    NaiveNUTSConfig,
    mcmc_naivenuts,
)
from parametricpinn.calibration.bayesian.prior import PriorFunc
from parametricpinn.calibration.bayesian.statistics import MomentsMultivariateNormal
from parametricpinn.calibration.data import CalibrationData, PreprocessedData
from parametricpinn.calibration.utility import freeze_model
from parametricpinn.errors import MCMCConfigError, UnvalidCalibrationDataError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.loaderssavers import PytorchModelLoader
from parametricpinn.types import Device, Module, NPArray

MCMC_Algorithm_Output: TypeAlias = tuple[MomentsMultivariateNormal, NPArray]
MCMC_Algorithm_Closure: TypeAlias = Callable[[], MCMC_Algorithm_Output]


def calibrate(
    model: Module,
    name_model_parameters_file: str,
    calibration_data: CalibrationData,
    prior: PriorFunc,
    mcmc_config: MCMCConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> MCMC_Algorithm_Output:
    preprocessed_data = _preprocess_calibration_data(calibration_data)
    model = _load_model(
        model=model,
        name_model_parameters_file=name_model_parameters_file,
        output_subdir=output_subdir,
        project_directory=project_directory,
        device=device,
    )
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
    prior: PriorFunc,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> MCMC_Algorithm_Closure:
    if isinstance(mcmc_config, MetropolisHastingsConfig):
        config_mh = cast(MetropolisHastingsConfig, mcmc_config)
        print("MCMC algorithm used: Metropolis Hastings")

        def mcmc_mh_algorithm() -> MCMC_Algorithm_Output:
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
        print("MCMC algorithm used: Hamiltonian")

        def mcmc_hamiltonian_algorithm() -> MCMC_Algorithm_Output:
            return mcmc_hamiltonian(
                parameter_names=config_h.parameter_names,
                true_parameters=config_h.true_parameters,
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_h.initial_parameters,
                num_leapfrog_steps=config_h.num_leabfrog_steps,
                leapfrog_step_sizes=config_h.leapfrog_step_sizes,
                num_iterations=config_h.num_iterations,
                num_burn_in_iterations=config_h.num_burn_in_iterations,
                output_subdir=output_subdir,
                project_directory=project_directory,
                device=device,
            )

        return mcmc_hamiltonian_algorithm
    elif isinstance(mcmc_config, NaiveNUTSConfig):
        config_nnuts = cast(NaiveNUTSConfig, mcmc_config)
        print("MCMC algorithm used: Naive NUTS")

        def mcmc_naive_nuts_algorithm() -> MCMC_Algorithm_Output:
            return mcmc_naivenuts(
                parameter_names=config_nnuts.parameter_names,
                true_parameters=config_nnuts.true_parameters,
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_nnuts.initial_parameters,
                leapfrog_step_sizes=config_nnuts.leapfrog_step_sizes,
                num_iterations=config_nnuts.num_iterations,
                num_burn_in_iterations=config_nnuts.num_burn_in_iterations,
                output_subdir=output_subdir,
                project_directory=project_directory,
                device=device,
            )

        return mcmc_naive_nuts_algorithm
    elif isinstance(mcmc_config, EfficientNUTSConfig):
        config_enuts = cast(EfficientNUTSConfig, mcmc_config)
        print("MCMC algorithm used: Efficient NUTS")

        def mcmc_efficient_nuts_algorithm() -> MCMC_Algorithm_Output:
            return mcmc_efficientnuts(
                parameter_names=config_enuts.parameter_names,
                true_parameters=config_enuts.true_parameters,
                likelihood=likelihood,
                prior=prior,
                initial_parameters=config_enuts.initial_parameters,
                leapfrog_step_sizes=config_enuts.leapfrog_step_sizes,
                num_iterations=config_enuts.num_iterations,
                num_burn_in_iterations=config_enuts.num_burn_in_iterations,
                output_subdir=output_subdir,
                project_directory=project_directory,
                device=device,
            )

        return mcmc_efficient_nuts_algorithm
    else:
        raise MCMCConfigError(
            f"There is no implementation for the requested MCMC algorithm {mcmc_config}."
        )
