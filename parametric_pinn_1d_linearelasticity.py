import os
from datetime import date
from time import perf_counter

import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_stretched_rod,
)
from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import create_univariate_uniform_distributed_prior
from parametricpinn.calibration import (
    CalibrationData,
    CalibrationDataGenerator1D,
    EfficientNUTSConfig,
    HamiltonianConfig,
    LeastSquaresConfig,
    MetropolisHastingsConfig,
    test_coverage,
    test_least_squares_calibration,
)
from parametricpinn.calibration.bayesianinference.likelihoods import (
    create_standard_ppinn_likelihood_for_noise,
    create_standard_ppinn_q_likelihood_for_noise,
)
from parametricpinn.calibration.utility import load_model
from parametricpinn.data.parameterssampling import sample_random, sample_uniform_grid
from parametricpinn.data.trainingdata_1d import (
    StretchedRodTrainingDataset1D,
    StretchedRodTrainingDataset1DConfig,
    create_training_dataset,
)
from parametricpinn.data.validationdata_linearelasticity_1d import (
    StretchedRodValidationDatasetLinearElasticity1D,
    StretchedRodValidationDatasetLinearElasticity1DConfig,
    calculate_linear_elastic_displacements_solution,
    create_validation_dataset,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig1D,
    plot_displacements_1d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_standard_linearelasticity_stretchedrod import (
    StandardTrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import NPArray, Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
num_material_parameters = 1
length = 100.0
traction = 100.0
volume_force = 0.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
displacement_left = 0.0
# Network
layer_sizes = [2, 16, 16, 16, 16, 1]
# Ansatz
distance_function = "normalized linear"
# Training
num_parameter_samples = 256
num_collocation_points = 128
training_batch_size = num_parameter_samples
number_training_epochs = 500
weight_pde_loss = 1.0
weight_traction_bc_loss = 1.0
# Validation
num_samples_valid = 1000
valid_interval = 1
num_points_valid = 512
batch_size_valid = num_samples_valid
# Calibration
consider_model_error = True
use_least_squares = True
use_random_walk_metropolis_hasting = True
use_hamiltonian = False
use_efficient_nuts = False
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_parametric_pinn_1D_E_{int(min_youngs_modulus)}_{int(max_youngs_modulus)}_samples_{num_parameter_samples}_col_{num_collocation_points}_neurons_4_16"
output_subdirectory_training = os.path.join(output_subdirectory, "training")


### Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def create_datasets() -> (
    tuple[
        StretchedRodTrainingDataset1D, StretchedRodValidationDatasetLinearElasticity1D
    ]
):
    def _create_training_dataset() -> StretchedRodTrainingDataset1D:
        parameter_samples = sample_uniform_grid(
            min_parameters=[min_youngs_modulus],
            max_parameters=[max_youngs_modulus],
            num_steps=[num_parameter_samples],
            device=device,
        )
        config_training_dataset = StretchedRodTrainingDataset1DConfig(
            parameters_samples=parameter_samples,
            length=length,
            traction=traction,
            volume_force=volume_force,
            num_points_pde=num_collocation_points,
        )
        return create_training_dataset(config_training_dataset)

    def _create_validation_dataset() -> (
        StretchedRodValidationDatasetLinearElasticity1DConfig
    ):
        offset_training_range_youngs_modulus = 1000.0

        config_validation_dataset = (
            StretchedRodValidationDatasetLinearElasticity1DConfig(
                length=length,
                min_youngs_modulus=min_youngs_modulus
                + offset_training_range_youngs_modulus,
                max_youngs_modulus=max_youngs_modulus
                - offset_training_range_youngs_modulus,
                traction=traction,
                volume_force=volume_force,
                num_points=num_points_valid,
                num_samples=num_samples_valid,
            )
        )
        return create_validation_dataset(config_validation_dataset)

    training_dataset = _create_training_dataset()
    validation_dataset = _create_validation_dataset()
    return training_dataset, validation_dataset


def create_ansatz() -> StandardAnsatz:
    def _determine_normalization_values() -> dict[str, Tensor]:
        min_coordinate = 0.0
        max_coordinate = length
        min_inputs = torch.tensor([min_coordinate, min_youngs_modulus])
        max_inputs = torch.tensor([max_coordinate, max_youngs_modulus])
        min_displacement = displacement_left
        max_displacement = calculate_linear_elastic_displacements_solution(
            coordinates=max_coordinate,
            length=length,
            youngs_modulus=min_youngs_modulus,
            traction=traction,
            volume_force=volume_force,
        )
        min_output = torch.tensor([min_displacement])
        max_output = torch.tensor([max_displacement])
        return {
            "min_inputs": min_inputs.to(device),
            "max_inputs": max_inputs.to(device),
            "min_output": min_output.to(device),
            "max_output": max_output.to(device),
        }

    normalization_values = _determine_normalization_values()
    network = FFNN(layer_sizes=layer_sizes)
    return create_standard_normalized_hbc_ansatz_stretched_rod(
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_output"],
        max_outputs=normalization_values["max_output"],
        network=network,
        distance_function_type=distance_function,
        device=device,
    ).to(device)


ansatz = create_ansatz()


def training_step() -> None:
    train_config = StandardTrainingConfiguration(
        ansatz=ansatz,
        weight_pde_loss=weight_pde_loss,
        weight_traction_bc_loss=weight_traction_bc_loss,
        training_dataset=training_dataset,
        number_training_epochs=number_training_epochs,
        training_batch_size=training_batch_size,
        validation_dataset=validation_dataset,
        output_subdirectory=output_subdirectory_training,
        project_directory=project_directory,
        device=device,
    )

    def _plot_exemplary_displacements() -> None:
        displacements_plotter_config = DisplacementsPlotterConfig1D()

        plot_displacements_1d(
            ansatz=ansatz,
            length=length,
            youngs_modulus_list=[min_youngs_modulus, 210000, max_youngs_modulus],
            traction=traction,
            volume_force=volume_force,
            output_subdir=output_subdirectory_training,
            project_directory=project_directory,
            config=displacements_plotter_config,
            device=device,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacements()


def calibration_step() -> None:
    print("Start calibration ...")
    num_data_points = 128
    std_noise = 5 * 1e-4
    num_test_cases = num_samples_valid

    initial_youngs_modulus = 210000.0
    prior = create_univariate_uniform_distributed_prior(
        lower_limit=min_youngs_modulus, upper_limit=max_youngs_modulus, device=device
    )

    parameter_names = ("Youngs modulus",)
    initial_parameters = torch.tensor([initial_youngs_modulus], device=device)

    def generate_calibration_data() -> tuple[tuple[CalibrationData, ...], NPArray]:
        true_parameters = sample_random(
            min_parameters=[min_youngs_modulus],
            max_parameters=[max_youngs_modulus],
            num_samples=num_test_cases,
            device=device,
        )
        calibration_data_generator = CalibrationDataGenerator1D(
            true_parameters=true_parameters,
            traction=traction,
            volume_force=volume_force,
            length=length,
            num_data_points=num_data_points,
            std_noise=std_noise,
            num_cases=num_test_cases,
            solution_func=calculate_linear_elastic_displacements_solution,
            device=device,
        )
        calibration_data_sets = calibration_data_generator.generate_data()
        return calibration_data_sets, true_parameters.detach().cpu().numpy()

    name_model_parameters_file = "model_parameters"
    model = load_model(
        model=ansatz,
        name_model_parameters_file=name_model_parameters_file,
        input_subdir=output_subdirectory_training,
        project_directory=project_directory,
        device=device,
    )

    calibration_data, true_parameters = generate_calibration_data()

    if consider_model_error:
        likelihoods = tuple(
            create_standard_ppinn_q_likelihood_for_noise(
                model=model,
                num_model_parameters=num_material_parameters,
                data=data,
                make_robust=False,
                device=device,
            )
            for data in calibration_data
        )
        output_subdir_calibration = os.path.join(
            output_subdirectory, "calibration_with_model_error"
        )
    else:
        likelihoods = tuple(
            create_standard_ppinn_likelihood_for_noise(
                model=model,
                num_model_parameters=num_material_parameters,
                data=data,
                device=device,
            )
            for data in calibration_data
        )
        output_subdir_calibration = os.path.join(
            output_subdirectory, "calibration_without_model_error"
        )

    def set_up_least_squares_configs(
        calibration_data: tuple[CalibrationData, ...],
    ) -> tuple[LeastSquaresConfig, ...]:
        configs = []
        for data in calibration_data:
            mean_displacements = torch.mean(torch.absolute(data.outputs), dim=0)
            residual_weights = (
                (1 / mean_displacements).to(device).repeat((num_data_points, 1)).ravel()
            )
            config = LeastSquaresConfig(
                ansatz=model,
                calibration_data=data,
                initial_parameters=initial_parameters,
                num_iterations=100,
                resdiual_weights=residual_weights,
            )
            configs.append(config)
        return tuple(configs)

    def set_up_metropolis_hastings_configs(
        likelihoods: tuple[Likelihood, ...],
    ) -> tuple[MetropolisHastingsConfig, ...]:
        configs = []
        for likelihood in likelihoods:
            std_proposal_density_youngs_modulus = 1000.0
            cov_proposal_density = torch.diag(
                torch.tensor(
                    [
                        std_proposal_density_youngs_modulus,
                    ],
                    dtype=torch.float64,
                    device=device,
                )
                ** 2
            )
            config = MetropolisHastingsConfig(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=initial_parameters,
                num_iterations=int(1e4),
                num_burn_in_iterations=int(5e3),
                cov_proposal_density=cov_proposal_density,
            )
            configs.append(config)
        return tuple(configs)

    def set_up_hamiltonian_configs(
        likelihoods: tuple[Likelihood, ...],
    ) -> tuple[HamiltonianConfig, ...]:
        configs = []
        for likelihood in likelihoods:
            config = HamiltonianConfig(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=initial_parameters,
                num_iterations=int(1e4),
                num_burn_in_iterations=int(1e3),
                num_leabfrog_steps=256,
                leapfrog_step_sizes=torch.tensor(1.0, device=device),
            )
            configs.append(config)
        return tuple(configs)

    def set_up_efficient_nuts_configs_configs(
        likelihoods: tuple[Likelihood, ...],
    ) -> tuple[EfficientNUTSConfig, ...]:
        configs = []
        for likelihood in likelihoods:
            config = EfficientNUTSConfig(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=initial_parameters,
                num_iterations=int(1e4),
                num_burn_in_iterations=int(1e3),
                max_tree_depth=8,
                leapfrog_step_sizes=torch.tensor(1.0, device=device),
            )
            configs.append(config)
        return tuple(configs)

    if use_least_squares:
        configs_ls = set_up_least_squares_configs(calibration_data)
        start = perf_counter()
        test_least_squares_calibration(
            calibration_configs=configs_ls,
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            output_subdir=os.path.join(output_subdir_calibration, "least_squares"),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time least squares test: {time}")
        print("############################################################")
    if use_random_walk_metropolis_hasting:
        configs_mh = set_up_metropolis_hastings_configs(likelihoods)
        start = perf_counter()
        test_coverage(
            calibration_configs=configs_mh,
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            output_subdir=os.path.join(
                output_subdir_calibration, "metropolis_hastings"
            ),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time Metropolis-Hasting coverage test: {time}")
        print("############################################################")
    if use_hamiltonian:
        configs_h = set_up_hamiltonian_configs(likelihoods)
        start = perf_counter()
        test_coverage(
            calibration_configs=configs_h,
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            output_subdir=os.path.join(output_subdir_calibration, "hamiltonian"),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time Hamiltonian coverage test: {time}")
        print("############################################################")
    if use_efficient_nuts:
        configs_en = set_up_efficient_nuts_configs_configs(likelihoods)
        start = perf_counter()
        test_coverage(
            calibration_configs=configs_en,
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            output_subdir=os.path.join(output_subdir_calibration, "efficient_nuts"),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time efficient NUTS coverage test: {time}")
        print("############################################################")
    print("Calibration finished.")


if retrain_parametric_pinn:
    training_dataset, validation_dataset = create_datasets()
    training_step()
calibration_step()
