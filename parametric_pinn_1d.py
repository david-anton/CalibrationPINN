from datetime import date

import torch

from parametricpinn.ansatz import create_normalized_hbc_ansatz_1D
from parametricpinn.calibration import (
    CalibrationData,
    EfficientNUTSConfig,
    HamiltonianConfig,
    MetropolisHastingsConfig,
    NaiveNUTSConfig,
    calibrate,
)
from parametricpinn.calibration.bayesian.prior import (
    compile_univariate_normal_distributed_prior,
)
from parametricpinn.data import (
    TrainingDataset1D,
    ValidationDataset1D,
    calculate_displacements_solution_1D,
    create_training_dataset_1D,
    create_validation_dataset_1D,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig1D,
    plot_displacements_1D,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_1D import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Module, Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
length = 100.0
traction = 1.0
volume_force = 1.0
min_youngs_modulus = 165000.0
max_youngs_modulus = 255000.0
displacement_left = 0.0
# Network
layer_sizes = [2, 16, 16, 1]
# Training
num_samples_train = 128
num_points_pde = 128
training_batch_size = num_samples_train
number_training_epochs = 200
# Validation
num_samples_valid = 128
valid_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
use_random_walk_metropolis_hasting = True
use_hamiltonian = True
use_naive_nuts = True
use_efficient_nuts = True
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_Parametric_PINN_1D"


### Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def create_datasets() -> tuple[TrainingDataset1D, ValidationDataset1D]:
    train_dataset = create_training_dataset_1D(
        length=length,
        traction=traction,
        volume_force=volume_force,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        num_points_pde=num_points_pde,
        num_samples=num_samples_train,
    )
    valid_dataset = create_validation_dataset_1D(
        length=length,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
        num_points=num_points_valid,
        num_samples=num_samples_valid,
    )
    return train_dataset, valid_dataset


def create_ansatz() -> Module:
    def _determine_normalization_values() -> dict[str, Tensor]:
        min_coordinate = 0.0
        max_coordinate = length
        min_inputs = torch.tensor([min_coordinate, min_youngs_modulus])
        max_inputs = torch.tensor([max_coordinate, max_youngs_modulus])
        min_displacement = displacement_left
        max_displacement = calculate_displacements_solution_1D(
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
    return create_normalized_hbc_ansatz_1D(
        displacement_left=torch.tensor([displacement_left]).to(device),
        network=network,
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_output"],
        max_outputs=normalization_values["max_output"],
    ).to(device)


ansatz = create_ansatz()


def training_step() -> None:
    train_config = TrainingConfiguration(
        ansatz=ansatz,
        training_dataset=training_dataset,
        number_training_epochs=number_training_epochs,
        training_batch_size=training_batch_size,
        validation_dataset=validation_dataset,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    def _plot_exemplary_displacements() -> None:
        displacements_plotter_config = DisplacementsPlotterConfig1D()

        plot_displacements_1D(
            ansatz=ansatz,
            length=length,
            youngs_modulus_list=[187634, 238695],
            traction=traction,
            volume_force=volume_force,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            config=displacements_plotter_config,
            device=device,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacements()


def calibration_step() -> None:
    print("Start calibration ...")
    exact_youngs_modulus = 195000
    num_points_calibration = 32
    std_noise = 5 * 1e-4

    def generate_calibration_data() -> tuple[Tensor, Tensor]:
        calibration_dataset = create_validation_dataset_1D(
            length=length,
            min_youngs_modulus=exact_youngs_modulus,
            max_youngs_modulus=exact_youngs_modulus,
            traction=traction,
            volume_force=volume_force,
            num_points=num_points_calibration,
            num_samples=1,
        )
        inputs, outputs = calibration_dataset[0]
        coordinates = torch.reshape(inputs[:, 0], (-1, 1)).to(device)
        clean_displacements = outputs.to(device)
        noise = torch.normal(
            mean=torch.tensor(0.0, device=device),
            std=torch.tensor(std_noise, device=device),
        )
        noisy_displacements = clean_displacements + noise
        return coordinates, noisy_displacements

    coordinates, noisy_displacements = generate_calibration_data()
    data = CalibrationData(
        inputs=coordinates,
        outputs=noisy_displacements,
        std_noise=std_noise,
    )

    prior_mean_youngs_modulus = 210000
    prior_std_youngs_modulus = 10000
    prior = compile_univariate_normal_distributed_prior(
        mean=prior_mean_youngs_modulus,
        standard_deviation=prior_std_youngs_modulus,
        device=device,
    )

    parameter_names = ("Youngs modulus",)
    true_parameters = (exact_youngs_modulus,)
    initial_parameters = torch.tensor([prior_mean_youngs_modulus], device=device)

    mcmc_config_mh = MetropolisHastingsConfig(
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e3),
        cov_proposal_density=torch.pow(torch.tensor([1000], device=device), 2),
    )
    mcmc_config_h = HamiltonianConfig(
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e4),
        num_leabfrog_steps=32,
        leapfrog_step_sizes=torch.tensor(1.0, device=device),
    )
    mcmc_config_nnuts = NaiveNUTSConfig(
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e3),
        leapfrog_step_sizes=torch.tensor(10.0, device=device),
    )
    mcmc_config_enuts = EfficientNUTSConfig(
        parameter_names=parameter_names,
        true_parameters=true_parameters,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e3),
        leapfrog_step_sizes=torch.tensor(10.0, device=device),
    )
    if use_random_walk_metropolis_hasting:
        posterior_moments_mh, samples_mh = calibrate(
            model=ansatz,
            name_model_parameters_file="model_parameters",
            calibration_data=data,
            prior=prior,
            mcmc_config=mcmc_config_mh,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )
    if use_hamiltonian:
        posterior_moments_h, samples_h = calibrate(
            model=ansatz,
            name_model_parameters_file="model_parameters",
            calibration_data=data,
            prior=prior,
            mcmc_config=mcmc_config_h,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )
    if use_naive_nuts:
        posterior_moments_nnuts, samples_nnuts = calibrate(
            model=ansatz,
            name_model_parameters_file="model_parameters",
            calibration_data=data,
            prior=prior,
            mcmc_config=mcmc_config_nnuts,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )
    if use_efficient_nuts:
        posterior_moments_enuts, samples_enuts = calibrate(
            model=ansatz,
            name_model_parameters_file="model_parameters",
            calibration_data=data,
            prior=prior,
            mcmc_config=mcmc_config_enuts,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            device=device,
        )
    print("Calibration finished.")


if retrain_parametric_pinn:
    training_dataset, validation_dataset = create_datasets()
    training_step()
calibration_step()
