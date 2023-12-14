import os
from datetime import date
from time import perf_counter

import numpy as np
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_clamped_left,
)
from parametricpinn.bayesian.prior import (
    create_univariate_normal_distributed_prior,
    multiply_priors,
)
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
from parametricpinn.data.trainingdata_elasticity_2d import (
    SimplifiedDogBoneGeometryConfig,
    SimplifiedDogBoneTrainingDataset2D,
    SimplifiedDogBoneTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.data.validationdata_elasticity_2d import (
    ValidationDataset2D,
    ValidationDataset2DConfig,
    create_validation_dataset,
)
from parametricpinn.fem import (
    LinearElasticityProblemConfig,
    SimplifiedDogBoneDomainConfig,
    SimulationConfig,
    generate_validation_data,
    run_simulation,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig2D,
    plot_displacements_2d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_standard_linearelasticity_dogbone import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
material_model = "plane stress"
num_material_parameters = 2
traction_right_x = 106.2629
traction_right_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 180000.0
max_youngs_modulus = min_youngs_modulus  # 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = min_poissons_ratio  # 0.4
# Network
layer_sizes = [4, 32, 32, 32, 32, 2]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_per_parameter = 1  # 32
num_collocation_points = 4096  # 128
number_points_per_bc = 128  # 32
bcs_overlap_angle_distance = 1e-6  # 1e-7
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 200  # 20000
weight_pde_loss = 1.0
weight_traction_bc_loss = 1.0
# Validation
regenerate_valid_data = True
input_subdir_valid = "20231211_validation_data_linearelasticity_simplifieddogbone_E_180k_nu_02_elementsize_01"  # "20231211_validation_data_linearelasticity_simplifieddogbone_E_180k_240k_nu_02_04_elementsize_01"
num_samples_valid = 1  # 32
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
input_subdir_calibration = "20231124_experimental_dic_data_dogbone"
input_file_name_calibration = "displacements.csv"
use_least_squares = False  # True
use_random_walk_metropolis_hasting = False  # True
use_hamiltonian = False  # True
use_efficient_nuts = False  # True
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size = 0.1
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_parametric_pinn_linearelasticity_simplifieddogbone_E_180k_nu_02_col_4096_bc_128_neurons_4_32_epochs_200"  # f"{output_date}_parametric_pinn_linearelasticity_simplifieddogbone_E_180k_240k_nu_02_04_samples_32_col_128_bc_32_neurons_4_32"
output_subdirectory_preprocessing = f"{output_date}_preprocessing"
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

geometry_config = SimplifiedDogBoneGeometryConfig()


def create_fem_domain_config() -> SimplifiedDogBoneDomainConfig:
    return SimplifiedDogBoneDomainConfig(
        geometry_config=geometry_config,
        traction_right_x=traction_right_x,
        traction_right_y=traction_right_y,
        element_family=fem_element_family,
        element_degree=fem_element_degree,
        element_size=fem_element_size,
    )


def create_datasets() -> tuple[SimplifiedDogBoneTrainingDataset2D, ValidationDataset2D]:
    def _create_training_dataset() -> SimplifiedDogBoneTrainingDataset2D:
        print("Generate training data ...")
        traction_right = torch.tensor([traction_right_x, traction_right_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        config_training_data = SimplifiedDogBoneTrainingDataset2DConfig(
            traction_right=traction_right,
            volume_force=volume_force,
            min_youngs_modulus=min_youngs_modulus,
            max_youngs_modulus=max_youngs_modulus,
            min_poissons_ratio=min_poissons_ratio,
            max_poissons_ratio=max_poissons_ratio,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=number_points_per_bc,
            num_samples_per_parameter=num_samples_per_parameter,
            bcs_overlap_angle_distance=bcs_overlap_angle_distance,
        )
        return create_training_dataset(config_training_data)

    def _create_validation_dataset() -> ValidationDataset2D:
        def _generate_validation_data() -> None:
            def _generate_random_parameter_list(
                size: int, min_value: float, max_value: float
            ) -> list[float]:
                random_params = min_value + torch.rand(size) * (max_value - min_value)
                return random_params.tolist()

            youngs_moduli = _generate_random_parameter_list(
                num_samples_valid, min_youngs_modulus, max_youngs_modulus
            )
            poissons_ratios = _generate_random_parameter_list(
                num_samples_valid, min_poissons_ratio, max_poissons_ratio
            )
            domain_config = create_fem_domain_config()
            problem_configs = []
            for i in range(num_samples_valid):
                problem_configs.append(
                    LinearElasticityProblemConfig(
                        model=material_model,
                        youngs_modulus=youngs_moduli[i],
                        poissons_ratio=poissons_ratios[i],
                    )
                )
            generate_validation_data(
                domain_config=domain_config,
                problem_configs=problem_configs,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
                save_metadata=save_metadata,
                output_subdir=input_subdir_valid,
                project_directory=project_directory,
            )

        print("Load validation data ...")
        if regenerate_valid_data:
            print("Run FE simulations to generate validation data ...")
            _generate_validation_data()
        config_validation_data = ValidationDataset2DConfig(
            input_subdir=input_subdir_valid,
            num_points=num_points_valid,
            num_samples=num_samples_valid,
            project_directory=project_directory,
        )
        return create_validation_dataset(config_validation_data)

    training_dataset = _create_training_dataset()
    validation_dataset = _create_validation_dataset()
    return training_dataset, validation_dataset


def create_ansatz() -> StandardAnsatz:
    def _determine_normalization_values() -> dict[str, Tensor]:
        min_coordinate_x = -geometry_config.left_half_box_length
        max_coordinate_x = geometry_config.right_half_box_length
        min_coordinate_y = -geometry_config.half_box_height
        max_coordinate_y = geometry_config.half_box_height
        min_coordinates = torch.tensor([min_coordinate_x, min_coordinate_y])
        max_coordinates = torch.tensor([max_coordinate_x, max_coordinate_y])

        min_parameters = torch.tensor([min_youngs_modulus, min_poissons_ratio])
        max_parameters = torch.tensor([max_youngs_modulus, max_poissons_ratio])

        min_inputs = torch.concat((min_coordinates, min_parameters))
        max_inputs = torch.concat((max_coordinates, max_parameters))

        _output_subdir = os.path.join(
            output_subdirectory_preprocessing,
            "results_for_determining_normalization_values",
        )
        print("Run FE simulation to determine normalization values ...")
        problem_config = LinearElasticityProblemConfig(
            model=material_model,
            youngs_modulus=min_youngs_modulus,
            poissons_ratio=max_poissons_ratio,
        )
        domain_config = create_fem_domain_config()
        simulation_config = SimulationConfig(
            domain_config=domain_config,
            problem_config=problem_config,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
        )
        simulation_results = run_simulation(
            simulation_config=simulation_config,
            save_results=False,
            save_metadata=False,
            output_subdir=_output_subdir,
            project_directory=project_directory,
        )

        min_displacement_x = float(np.amin(simulation_results.displacements_x))
        max_displacement_x = float(np.amax(simulation_results.displacements_x))
        min_displacement_y = float(np.amin(simulation_results.displacements_y))
        max_displacement_y = float(np.amax(simulation_results.displacements_y))
        min_outputs = torch.tensor([min_displacement_x, min_displacement_y])
        max_outputs = torch.tensor([max_displacement_x, max_displacement_y])
        return {
            "min_inputs": min_inputs.to(device),
            "max_inputs": max_inputs.to(device),
            "min_outputs": min_outputs.to(device),
            "max_outputs": max_outputs.to(device),
        }

    normalization_values = _determine_normalization_values()
    network = FFNN(layer_sizes=layer_sizes)
    return create_standard_normalized_hbc_ansatz_clamped_left(
        coordinate_x_left=torch.tensor(
            [-geometry_config.left_half_box_length], device=device
        ),
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_outputs"],
        max_outputs=normalization_values["max_outputs"],
        network=network,
        distance_function_type=distance_function,
        device=device,
    ).to(device)


ansatz = create_ansatz()


def training_step() -> None:
    train_config = TrainingConfiguration(
        ansatz=ansatz,
        material_model=material_model,
        weight_pde_loss=weight_pde_loss,
        weight_traction_bc_loss=weight_traction_bc_loss,
        training_dataset=training_dataset,
        number_training_epochs=number_training_epochs,
        training_batch_size=training_batch_size,
        validation_dataset=validation_dataset,
        validation_interval=validation_interval,
        output_subdirectory=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    def _plot_exemplary_displacement_fields() -> None:
        displacements_plotter_config = DisplacementsPlotterConfig2D()
        youngs_modulus_and_poissons_ratio_list = [
            (min_youngs_modulus, min_poissons_ratio),
            (min_youngs_modulus, max_poissons_ratio),
            (max_youngs_modulus, min_poissons_ratio),
            (max_youngs_modulus, max_poissons_ratio),
            (210000, 0.3),
        ]
        youngs_moduli, poissons_ratios = zip(*youngs_modulus_and_poissons_ratio_list)

        domain_config = create_fem_domain_config()
        problem_configs = []
        for i in range(len(youngs_modulus_and_poissons_ratio_list)):
            problem_configs.append(
                LinearElasticityProblemConfig(
                    model=material_model,
                    youngs_modulus=youngs_moduli[i],
                    poissons_ratio=poissons_ratios[i],
                )
            )

        plot_displacements_2d(
            ansatz=ansatz,
            domain_config=domain_config,
            problem_configs=problem_configs,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            plot_config=displacements_plotter_config,
            device=device,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacement_fields()


def calibration_step() -> None:
    print("Start calibration ...")
    exact_youngs_modulus = 218454.0
    exact_poissons_ratio = 0.2513
    num_data_points = 128
    std_noise = 5 * 1e-4

    def generate_calibration_data() -> tuple[Tensor, Tensor]:
        csv_reader = CSVDataReader(project_directory)
        data = csv_reader.read(
            file_name=input_file_name_calibration, subdir_name=input_subdir_calibration
        )
        size_data = len(data)
        slice_coordinates = slice(0, 2)
        slice_displacements = slice(2, 4)
        full_shifted_coordinates = torch.from_numpy(data[:, slice_coordinates]).type(
            torch.float64
        )
        coordinate_shift_x = geometry_config.left_half_parallel_length
        coordinate_shift_y = geometry_config.half_parallel_height
        full_coordinates = full_shifted_coordinates - torch.tensor(
            [coordinate_shift_x, coordinate_shift_y], dtype=torch.float64
        )
        full_displacements = torch.from_numpy(data[:, slice_displacements]).type(
            torch.float64
        )
        random_indices = torch.randint(
            low=0, high=size_data + 1, size=(num_data_points,)
        )
        coordinates = full_coordinates[random_indices, :].to(device)
        displacements = full_displacements[random_indices, :].to(device)
        return coordinates, displacements

    name_model_parameters_file = "model_parameters"
    model = load_model(
        model=ansatz,
        name_model_parameters_file=name_model_parameters_file,
        input_subdir=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    coordinates, displacements = generate_calibration_data()
    data = CalibrationData(
        inputs=coordinates,
        outputs=displacements,
        std_noise=std_noise,
    )
    likelihood = create_standard_ppinn_likelihood_for_noise(
        model=model,
        data=data,
        num_model_parameters=num_material_parameters,
        device=device,
    )

    prior_mean_youngs_modulus = 210000
    prior_std_youngs_modulus = 10000
    prior_mean_poissons_ratio = 0.3
    prior_std_poissons_ratio = 0.015
    prior_youngs_modulus = create_univariate_normal_distributed_prior(
        mean=prior_mean_youngs_modulus,
        standard_deviation=prior_std_youngs_modulus,
        device=device,
    )
    prior_poissons_ratio = create_univariate_normal_distributed_prior(
        mean=prior_mean_poissons_ratio,
        standard_deviation=prior_std_poissons_ratio,
        device=device,
    )
    prior = multiply_priors([prior_youngs_modulus, prior_poissons_ratio])

    parameter_names = ("Youngs modulus", "Poissons ratio")
    true_parameters = (exact_youngs_modulus, exact_poissons_ratio)
    initial_parameters = torch.tensor(
        [prior_mean_youngs_modulus, prior_mean_poissons_ratio], device=device
    )

    least_squares_config = LeastSquaresConfig(
        initial_parameters=initial_parameters,
        num_iterations=100,
        ansatz=model,
        calibration_data=data,
    )
    std_proposal_density_youngs_modulus = 1000
    std_proposal_density_poissons_ratio = 0.0015
    mcmc_config_mh = MetropolisHastingsConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(5e3),
        cov_proposal_density=torch.diag(
            torch.tensor(
                [
                    std_proposal_density_youngs_modulus,
                    std_proposal_density_poissons_ratio,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        ),
    )
    mcmc_config_h = HamiltonianConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(5e3),
        num_burn_in_iterations=int(1e3),
        num_leabfrog_steps=256,
        leapfrog_step_sizes=torch.tensor([1, 0.001], device=device),
    )
    mcmc_config_enuts = EfficientNUTSConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e3),
        num_burn_in_iterations=int(1e3),
        max_tree_depth=8,
        leapfrog_step_sizes=torch.tensor([1, 0.001], device=device),
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
