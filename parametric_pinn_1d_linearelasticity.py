from datetime import date
from time import perf_counter

import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_stretched_rod,
)
from parametricpinn.bayesian.prior import create_univariate_normal_distributed_prior
from parametricpinn.calibration import (
    CalibrationData,
    EfficientNUTSConfig,
    HamiltonianConfig,
    LeastSquaresConfig,
    MetropolisHastingsConfig,
    calibrate,
)
from parametricpinn.calibration.bayesianinference.parametric_pinn import (
    create_standard_ppinn_likelihood_for_noise,
)
from parametricpinn.calibration.bayesianinference.plot import (
    plot_posterior_normal_distributions,
)
from parametricpinn.calibration.utility import load_model
from parametricpinn.data.parameterssampling import sample_uniform_grid
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
from parametricpinn.types import Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
num_model_parameters = 1
length = 100.0
traction = 1.0
volume_force = 1.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
displacement_left = 0.0
# Network
layer_sizes = [2, 16, 16, 16, 16, 1]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_train = 128
num_points_pde = 128
training_batch_size = num_samples_train
number_training_epochs = 300
# Validation
num_samples_valid = 64
valid_interval = 1
num_points_valid = 512
batch_size_valid = num_samples_valid
# Calibration
use_least_squares = True
use_random_walk_metropolis_hasting = True
use_hamiltonian = False
use_efficient_nuts = False
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


def create_datasets() -> (
    tuple[
        StretchedRodTrainingDataset1D, StretchedRodValidationDatasetLinearElasticity1D
    ]
):
    parameter_samples = sample_uniform_grid(
        min_parameters=[min_youngs_modulus],
        max_parameters=[max_youngs_modulus],
        num_steps=[num_samples_train],
        device=device,
    )
    config_training_dataset = StretchedRodTrainingDataset1DConfig(
        parameters_samples=parameter_samples,
        length=length,
        traction=traction,
        volume_force=volume_force,
        num_points_pde=num_points_pde,
    )
    train_dataset = create_training_dataset(config_training_dataset)
    config_validation_dataset = StretchedRodValidationDatasetLinearElasticity1DConfig(
        length=length,
        min_youngs_modulus=min_youngs_modulus,
        max_youngs_modulus=max_youngs_modulus,
        traction=traction,
        volume_force=volume_force,
        num_points=num_points_valid,
        num_samples=num_samples_valid,
    )
    valid_dataset = create_validation_dataset(config_validation_dataset)
    return train_dataset, valid_dataset


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

        plot_displacements_1d(
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
        config_validation_dataset = (
            StretchedRodValidationDatasetLinearElasticity1DConfig(
                length=length,
                min_youngs_modulus=exact_youngs_modulus,
                max_youngs_modulus=exact_youngs_modulus,
                traction=traction,
                volume_force=volume_force,
                num_points=num_points_calibration,
                num_samples=1,
            )
        )
        calibration_dataset = create_validation_dataset(config_validation_dataset)
        inputs, outputs = calibration_dataset[0]
        coordinates = torch.reshape(inputs[:, 0], (-1, 1)).to(device)
        clean_displacements = outputs.to(device)
        noise = torch.normal(
            mean=torch.tensor(0.0, device=device),
            std=torch.tensor(std_noise, device=device),
        )
        noisy_displacements = clean_displacements + noise
        return coordinates, noisy_displacements

    name_model_parameters_file = "model_parameters"
    model = load_model(
        model=ansatz,
        name_model_parameters_file=name_model_parameters_file,
        input_subdir=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    coordinates, noisy_displacements = generate_calibration_data()
    data = CalibrationData(
        inputs=coordinates,
        outputs=noisy_displacements,
        std_noise=std_noise,
    )
    likelihood = create_standard_ppinn_likelihood_for_noise(
        model=model, num_model_parameters=num_model_parameters, data=data, device=device
    )

    prior_mean_youngs_modulus = 210000
    prior_std_youngs_modulus = 10000
    prior = create_univariate_normal_distributed_prior(
        mean=prior_mean_youngs_modulus,
        standard_deviation=prior_std_youngs_modulus,
        device=device,
    )

    parameter_names = ("Youngs modulus",)
    true_parameters = (exact_youngs_modulus,)
    initial_parameters = torch.tensor([prior_mean_youngs_modulus], device=device)

    least_squares_config = LeastSquaresConfig(
        initial_parameters=initial_parameters,
        num_iterations=100,
        ansatz=model,
        calibration_data=data,
    )
    mcmc_config_mh = MetropolisHastingsConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e3),
        cov_proposal_density=torch.pow(torch.tensor([1000], device=device), 2),
    )
    mcmc_config_h = HamiltonianConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e3),
        num_leabfrog_steps=256,
        leapfrog_step_sizes=torch.tensor(1.0, device=device),
    )
    mcmc_config_enuts = EfficientNUTSConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e3),
        max_tree_depth=8,
        leapfrog_step_sizes=torch.tensor(10.0, device=device),
    )
    if use_least_squares:
        start = perf_counter()
        identified_parameters, _ = calibrate(
            calibration_config=least_squares_config,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Identified parameter: {identified_parameters}")
        print(f"Run time least squares: {time}")
    if use_random_walk_metropolis_hasting:
        start = perf_counter()
        posterior_moments_mh, samples_mh = calibrate(
            calibration_config=mcmc_config_mh,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time Metropolis-Hasting: {time}")
        plot_posterior_normal_distributions(
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            moments=posterior_moments_mh,
            samples=samples_mh,
            mcmc_algorithm="metropolis_hastings",
            output_subdir=output_subdirectory,
            project_directory=project_directory,
        )
    if use_hamiltonian:
        start = perf_counter()
        posterior_moments_h, samples_h = calibrate(
            calibration_config=mcmc_config_h,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time Hamiltonian: {time}")
        plot_posterior_normal_distributions(
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            moments=posterior_moments_h,
            samples=samples_h,
            mcmc_algorithm="hamiltonian",
            output_subdir=output_subdirectory,
            project_directory=project_directory,
        )
    if use_efficient_nuts:
        start = perf_counter()
        posterior_moments_enuts, samples_enuts = calibrate(
            calibration_config=mcmc_config_enuts,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time efficient NUTS: {time}")
        plot_posterior_normal_distributions(
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            moments=posterior_moments_enuts,
            samples=samples_enuts,
            mcmc_algorithm="efficient_nuts",
            output_subdir=output_subdirectory,
            project_directory=project_directory,
        )
    print("Calibration finished.")


if retrain_parametric_pinn:
    training_dataset, validation_dataset = create_datasets()
    training_step()
calibration_step()
