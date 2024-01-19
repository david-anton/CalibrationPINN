import os
from datetime import date
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_quarter_plate_with_hole,
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
    create_standard_ppinn_likelihood_for_noise_and_model_error,
)
from parametricpinn.calibration.bayesianinference.plot import (
    plot_posterior_normal_distributions,
)
from parametricpinn.calibration.utility import load_model
from parametricpinn.data.parameterssampling import sample_uniform_grid
from parametricpinn.data.trainingdata_2d import (
    QuarterPlateWithHoleTrainingDataset2D,
    QuarterPlateWithHoleTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.data.validationdata_2d import (
    ValidationDataset2D,
    ValidationDataset2DConfig,
    create_validation_dataset,
)
from parametricpinn.fem import (
    NeoHookeanProblemConfig,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    generate_validation_data,
    run_simulation,
)
from parametricpinn.gps import IndependentMultiOutputGP, ZeroMeanScaledRBFKernelGP
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader, PandasDataWriter
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig2D,
    plot_displacements_2d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_standard_neohookean_quarterplatewithhole import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
num_material_parameters = 2
edge_length = 100.0
radius = 10.0
traction_left_x = -10.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_bulk_modulus = 1000.0
max_bulk_modulus = 2000.0
min_rivlin_saunders_c_10 = 10.0
max_rivlin_saunders_c_10 = 30.0
# Network
layer_sizes = [4, 64, 64, 64, 64, 2]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_per_parameter = 1  # 32
num_collocation_points = 128
number_points_per_bc = 64
bcs_overlap_distance = 1e-2
bcs_overlap_angle_distance = 1e-2
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 1  # 20000
weight_pde_loss = 1.0
weight_stress_bc_loss = 1.0
weight_traction_bc_loss = 1.0
# Validation
regenerate_valid_data = True
input_subdir_valid = "20240119_validation_data_neohookean_quarterplatewithhole_K_1000_2000_c10_10_30_edge_100_radius_10_traction_10_elementsize_02"
num_samples_valid = 1  # 32
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
consider_model_error = False
use_least_squares = False  # True
use_random_walk_metropolis_hasting = False  # True
use_hamiltonian = False
use_efficient_nuts = False
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 2
fem_element_size = 0.2
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_parametric_pinn_neohookean_quarterplatewithhole_K_1000_2000_c10_10_30_col_128_bc_64_neurons_4_64_traction_10"
output_subdirectory_preprocessing = f"{output_date}_preprocessing"
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def create_fem_domain_config() -> QuarterPlateWithHoleDomainConfig:
    return QuarterPlateWithHoleDomainConfig(
        edge_length=edge_length,
        radius=radius,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        element_family=fem_element_family,
        element_degree=fem_element_degree,
        element_size=fem_element_size,
    )


def create_datasets() -> (
    tuple[QuarterPlateWithHoleTrainingDataset2D, ValidationDataset2D]
):
    def _create_training_dataset() -> QuarterPlateWithHoleTrainingDataset2D:
        print("Generate training data ...")
        parameters_samples = sample_uniform_grid(
            min_parameters=[min_bulk_modulus, min_rivlin_saunders_c_10],
            max_parameters=[max_bulk_modulus, max_rivlin_saunders_c_10],
            num_steps=[num_samples_per_parameter, num_samples_per_parameter],
            device=device,
        )
        traction_left = torch.tensor([traction_left_x, traction_left_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        config_training_data = QuarterPlateWithHoleTrainingDataset2DConfig(
            parameters_samples=parameters_samples,
            edge_length=edge_length,
            radius=radius,
            traction_left=traction_left,
            volume_force=volume_force,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=number_points_per_bc,
            bcs_overlap_distance=bcs_overlap_distance,
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

            bulk_moduli = _generate_random_parameter_list(
                num_samples_valid, min_bulk_modulus, max_bulk_modulus
            )
            rivlin_saunders_c_10s = _generate_random_parameter_list(
                num_samples_valid, min_rivlin_saunders_c_10, max_rivlin_saunders_c_10
            )
            domain_config = create_fem_domain_config()
            problem_configs = []
            for i in range(num_samples_valid):
                problem_configs.append(
                    NeoHookeanProblemConfig(
                        material_parameters=(bulk_moduli[i], rivlin_saunders_c_10s[i]),
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
    normalization_values_subdir = os.path.join(
        output_subdirectory, "normalization_values"
    )
    key_min_inputs = "min_inputs"
    key_max_inputs = "max_inputs"
    key_min_outputs = "min_outputs"
    key_max_outputs = "max_outputs"
    file_name_min_inputs = "minimum_inputs.csv"
    file_name_max_inputs = "maximum_inputs.csv"
    file_name_min_outputs = "minimum_outputs.csv"
    file_name_max_outputs = "maximum_outputs.csv"

    def _save_normalization_values(normalization_values: dict[str, Tensor]) -> None:
        data_writer = PandasDataWriter(project_directory)

        def _save_one_value_tensor(key: str, file_name: str) -> None:
            data_writer.write(
                data=pd.DataFrame([normalization_values[key].cpu().detach().numpy()]),
                file_name=file_name,
                subdir_name=normalization_values_subdir,
                header=True,
            )

        _save_one_value_tensor(key_min_inputs, file_name_min_inputs)
        _save_one_value_tensor(key_max_inputs, file_name_max_inputs)
        _save_one_value_tensor(key_min_outputs, file_name_min_outputs)
        _save_one_value_tensor(key_max_outputs, file_name_max_outputs)

    def _read_normalization_values() -> dict[str, Tensor]:
        data_reader = CSVDataReader(project_directory)
        normalization_values = {}

        def _add_one_value_tensor(key: str, file_name: str) -> None:
            values = data_reader.read(
                file_name=file_name,
                subdir_name=normalization_values_subdir,
                read_from_output_dir=True,
            )

            normalization_values[key] = (
                torch.from_numpy(values[0]).type(torch.float64).to(device)
            )

        _add_one_value_tensor(key_min_inputs, file_name_min_inputs)
        _add_one_value_tensor(key_max_inputs, file_name_max_inputs)
        _add_one_value_tensor(key_min_outputs, file_name_min_outputs)
        _add_one_value_tensor(key_max_outputs, file_name_max_outputs)

        return normalization_values

    def _print_normalization_values(normalization_values: dict[str, Tensor]) -> None:
        print("###########################")
        print(f"Min inputs {normalization_values[key_min_inputs]}")
        print(f"Max inputs {normalization_values[key_max_inputs]}")
        print(f"Min outputs {normalization_values[key_min_outputs]}")
        print(f"Max outputs {normalization_values[key_max_outputs]}")
        print("###########################")

    def _determine_normalization_values() -> dict[str, Tensor]:
        if retrain_parametric_pinn:
            min_coordinate_x = -edge_length
            max_coordinate_x = 0.0
            min_coordinate_y = 0.0
            max_coordinate_y = edge_length
            min_coordinates = torch.tensor([min_coordinate_x, min_coordinate_y])
            max_coordinates = torch.tensor([max_coordinate_x, max_coordinate_y])

            min_parameters = torch.tensor([min_bulk_modulus, min_rivlin_saunders_c_10])
            max_parameters = torch.tensor([max_bulk_modulus, max_rivlin_saunders_c_10])

            min_inputs = torch.concat((min_coordinates, min_parameters))
            max_inputs = torch.concat((max_coordinates, max_parameters))

            print("Run FE simulations to determine normalization values ...")
            domain_config = create_fem_domain_config()
            problem_config = NeoHookeanProblemConfig(
                material_parameters=(min_bulk_modulus, max_rivlin_saunders_c_10)
            )
            simulation_config = SimulationConfig(
                domain_config=domain_config,
                problem_config=problem_config,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
            )
            simulation_results = run_simulation(
                simulation_config=simulation_config,
                save_results=True,
                save_metadata=True,
                output_subdir=normalization_values_subdir,
                project_directory=project_directory,
            )

            min_displacement_x = float(np.amin(simulation_results.displacements_x))
            max_displacement_x = float(np.amax(simulation_results.displacements_x))
            min_displacement_y = float(np.amin(simulation_results.displacements_y))
            max_displacement_y = float(np.amax(simulation_results.displacements_y))
            min_outputs = torch.tensor([min_displacement_x, min_displacement_y])
            max_outputs = torch.tensor([max_displacement_x, max_displacement_y])
            normalization_values = {
                "min_inputs": min_inputs.to(device),
                "max_inputs": max_inputs.to(device),
                "min_outputs": min_outputs.to(device),
                "max_outputs": max_outputs.to(device),
            }
            _save_normalization_values(normalization_values)
        else:
            normalization_values = _read_normalization_values()
        _print_normalization_values(normalization_values)
        return normalization_values

    normalization_values = _determine_normalization_values()
    network = FFNN(layer_sizes=layer_sizes)
    return create_standard_normalized_hbc_ansatz_quarter_plate_with_hole(
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
        number_points_per_bc=number_points_per_bc,
        weight_pde_loss=weight_pde_loss,
        weight_stress_bc_loss=weight_stress_bc_loss,
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
        parameters_list = [
            (min_bulk_modulus, min_rivlin_saunders_c_10),
            # (min_bulk_modulus, max_rivlin_saunders_c_10),
            # (max_bulk_modulus, min_rivlin_saunders_c_10),
            # (max_bulk_modulus, max_rivlin_saunders_c_10),
        ]
        bulk_moduli, rivlin_saunders_c_10s = zip(*parameters_list)

        domain_config = create_fem_domain_config()
        problem_configs = []
        for i in range(len(parameters_list)):
            problem_configs.append(
                NeoHookeanProblemConfig(
                    material_parameters=(bulk_moduli[i], rivlin_saunders_c_10s[i])
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
    exact_bulk_modulus = 1200
    exact_rivlin_saunders_c_10 = 15
    num_data_points = 128
    std_noise = 5 * 1e-4

    def generate_calibration_data() -> tuple[Tensor, Tensor]:
        domain_config = create_fem_domain_config()
        problem_config = NeoHookeanProblemConfig(
            material_parameters=(exact_bulk_modulus, exact_rivlin_saunders_c_10)
        )
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
            output_subdir=output_subdirectory,
            project_directory=project_directory,
        )
        total_size_data = simulation_results.coordinates_x.shape[0]
        random_indices = torch.randint(
            low=0, high=total_size_data + 1, size=(num_data_points,)
        )
        coordinates_x = torch.tensor(simulation_results.coordinates_x)[random_indices]
        coordinates_y = torch.tensor(simulation_results.coordinates_y)[random_indices]
        coordinates = torch.concat((coordinates_x, coordinates_y), dim=1).to(device)
        clean_displacements_x = torch.tensor(simulation_results.displacements_x)[
            random_indices
        ]
        clean_displacements_y = torch.tensor(simulation_results.displacements_y)[
            random_indices
        ]
        noisy_displacements_x = clean_displacements_x + torch.normal(
            mean=0.0, std=std_noise, size=clean_displacements_x.size()
        )
        noisy_displacements_y = clean_displacements_y + torch.normal(
            mean=0.0, std=std_noise, size=clean_displacements_y.size()
        )
        noisy_displacements = torch.concat(
            (noisy_displacements_x, noisy_displacements_y), dim=1
        ).to(device)
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

    prior_mean_youngs_modulus = 2000
    prior_std_youngs_modulus = 100
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
    initial_material_parameters = torch.tensor(
        [prior_mean_youngs_modulus, prior_mean_poissons_ratio], device=device
    )

    if consider_model_error:
        model_error_gp = IndependentMultiOutputGP(
            independent_gps=[
                ZeroMeanScaledRBFKernelGP(device),
                ZeroMeanScaledRBFKernelGP(device),
            ],
            device=device,
        ).to(device)
        likelihood = create_standard_ppinn_likelihood_for_noise_and_model_error(
            model=model,
            num_model_parameters=num_material_parameters,
            model_error_gp=model_error_gp,
            data=data,
            device=device,
        )

        model_error_prior = model_error_gp.get_uninformed_parameters_prior(
            device,
            upper_limit_output_scale=2.0,
            upper_limit_length_scale=2.0,
        )
        prior = multiply_priors(
            [prior_youngs_modulus, prior_poissons_ratio, model_error_prior]
        )

        parameter_names = (
            "Youngs modulus",
            "Poissons ratio",
            "error output scale 1",
            "error length scale 1",
            "error output scale 2",
            "error length scale 2",
        )
        true_parameters = (
            exact_bulk_modulus,
            exact_rivlin_saunders_c_10,
            None,
            None,
            None,
            None,
        )
        initial_gp_output_scale = 1.0
        initial_gp_length_scale = 1.0
        initial_model_error_parameters = torch.tensor(
            [
                initial_gp_output_scale,
                initial_gp_length_scale,
                initial_gp_output_scale,
                initial_gp_length_scale,
            ],
            dtype=torch.float64,
            device=device,
        )
        initial_parameters = torch.concat(
            [initial_material_parameters, initial_model_error_parameters]
        )
    else:
        likelihood = create_standard_ppinn_likelihood_for_noise(
            model=model,
            num_model_parameters=num_material_parameters,
            data=data,
            device=device,
        )
        prior = multiply_priors([prior_youngs_modulus, prior_poissons_ratio])

        parameter_names = ("Bulk modulus", "Rivlin-Saunders c_10")
        true_parameters = (exact_bulk_modulus, exact_rivlin_saunders_c_10)
        initial_parameters = initial_material_parameters

    least_squares_config = LeastSquaresConfig(
        initial_parameters=initial_material_parameters,
        num_iterations=1000,
        ansatz=model,
        calibration_data=data,
    )
    std_proposal_density_youngs_modulus = 1.0
    std_proposal_density_poissons_ratio = 1.5 * 1e-4
    if consider_model_error:
        std_proposal_density_gp_output_scale = 1e-3
        std_proposal_density_gp_length_scale = 1e-3
        cov_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_youngs_modulus,
                    std_proposal_density_poissons_ratio,
                    std_proposal_density_gp_output_scale,
                    std_proposal_density_gp_length_scale,
                    std_proposal_density_gp_output_scale,
                    std_proposal_density_gp_length_scale,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )
    else:
        cov_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_youngs_modulus,
                    std_proposal_density_poissons_ratio,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )

    mcmc_config_mh = MetropolisHastingsConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e5),
        num_burn_in_iterations=int(1e5),
        cov_proposal_density=cov_proposal_density,
    )
    mcmc_config_h = HamiltonianConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e4),
        num_leabfrog_steps=256,
        leapfrog_step_sizes=torch.tensor([1, 0.01], device=device),
    )
    mcmc_config_enuts = EfficientNUTSConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e4),
        max_tree_depth=8,
        leapfrog_step_sizes=torch.tensor([1, 0.01], device=device),
    )
    if consider_model_error:
        output_subdir_calibration = os.path.join(
            output_subdirectory, "calibration_with_model_error"
        )
    else:
        output_subdir_calibration = os.path.join(
            output_subdirectory, "calibration_without_model_error"
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
        print("############################################################")
    if use_random_walk_metropolis_hasting:
        start = perf_counter()
        posterior_moments_mh, samples_mh = calibrate(
            calibration_config=mcmc_config_mh,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Identified moments: {posterior_moments_mh}")
        print(f"Run time Metropolis-Hasting: {time}")
        print("############################################################")
        plot_posterior_normal_distributions(
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            moments=posterior_moments_mh,
            samples=samples_mh,
            mcmc_algorithm="metropolis_hastings",
            output_subdir=output_subdir_calibration,
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
        print(f"Identified moments: {posterior_moments_h}")
        print(f"Run time Hamiltonian: {time}")
        print("############################################################")
        plot_posterior_normal_distributions(
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            moments=posterior_moments_h,
            samples=samples_h,
            mcmc_algorithm="hamiltonian",
            output_subdir=output_subdir_calibration,
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
        print(f"Identified moments: {posterior_moments_enuts}")
        print(f"Run time efficient NUTS: {time}")
        print("############################################################")
        plot_posterior_normal_distributions(
            parameter_names=parameter_names,
            true_parameters=true_parameters,
            moments=posterior_moments_enuts,
            samples=samples_enuts,
            mcmc_algorithm="efficient_nuts",
            output_subdir=output_subdir_calibration,
            project_directory=project_directory,
        )
    print("Calibration finished.")


if retrain_parametric_pinn:
    training_dataset, validation_dataset = create_datasets()
    training_step()
calibration_step()
