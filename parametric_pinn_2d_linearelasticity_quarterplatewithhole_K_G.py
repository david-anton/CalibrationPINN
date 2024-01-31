import os
from datetime import date
from time import perf_counter

import numpy as np
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
    LinearElasticityProblemConfig_K_G,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    generate_validation_data,
    run_simulation,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig2D,
    plot_displacements_2d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.loss_2d.momentum_linearelasticity_K_G import (
    calculate_E_from_K_and_G_factory,
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
    calculate_nu_from_K_and_G_factory,
)
from parametricpinn.training.training_standard_linearelasticity_quarterplatewithhole_K_G import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
material_model = "plane stress"
num_material_parameters = 2
edge_length = 100.0
radius = 10.0
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 160000.0
max_youngs_modulus = 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = 0.4
# Network
layer_sizes = [4, 64, 64, 64, 64, 2]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_per_parameter = 32
num_collocation_points = 128
num_points_per_bc = 64
bcs_overlap_distance = 1e-2
bcs_overlap_angle_distance = 1e-2
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 10000
weight_pde_loss = 1.0
weight_stress_bc_loss = 1.0
weight_traction_bc_loss = 1.0
# Validation
regenerate_valid_data = False
input_subdir_valid = "20240126_validation_data_linearelasticity_quarterplatewithhole_E_160k_240k_nu_02_04_edge_100_radius_10_traction_100_elementsize_01_K_G"
num_samples_valid = 32
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
use_least_squares = True
use_random_walk_metropolis_hasting = True
use_hamiltonian = True
use_efficient_nuts = True
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size = 0.1
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_parametric_pinn_linearelasticity_quarterplatewithhole_E_160k_240k_nu_02_04_samples_32_col_128_bc_64_neurons_4_64_K_G"
output_subdirectory_preprocessing = f"{output_date}_preprocessing"
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


calculate_K_from_E_and_nu = calculate_K_from_E_and_nu_factory(material_model)
calculate_E_from_K_and_G = calculate_E_from_K_and_G_factory(material_model)
calculate_nu_from_K_and_G = calculate_nu_from_K_and_G_factory(material_model)
min_bulk_modulus = calculate_K_from_E_and_nu(
    E=min_youngs_modulus, nu=min_poissons_ratio
)
max_bulk_modulus = calculate_K_from_E_and_nu(
    E=max_youngs_modulus, nu=max_poissons_ratio
)
min_shear_modulus = calculate_G_from_E_and_nu(
    E=min_youngs_modulus, nu=max_poissons_ratio
)
max_shear_modulus = calculate_G_from_E_and_nu(
    E=max_youngs_modulus, nu=min_poissons_ratio
)


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
        parameters_samples_E_nu = sample_uniform_grid(
            min_parameters=[min_youngs_modulus, min_poissons_ratio],
            max_parameters=[max_youngs_modulus, max_poissons_ratio],
            num_steps=[num_samples_per_parameter, num_samples_per_parameter],
            device=device,
        )
        parameter_samples_E = parameters_samples_E_nu[:, 0]
        parameter_samples_nu = parameters_samples_E_nu[:, 1]
        parameter_samples_K = calculate_K_from_E_and_nu(
            E=parameter_samples_E, nu=parameter_samples_nu
        ).reshape((-1, 1))
        parameter_samples_G = calculate_G_from_E_and_nu(
            E=parameter_samples_E, nu=parameter_samples_nu
        ).reshape((-1, 1))
        parameters_samples_K_G = torch.concat(
            (parameter_samples_K, parameter_samples_G), dim=1
        )
        traction_left = torch.tensor([traction_left_x, traction_left_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        config_training_data = QuarterPlateWithHoleTrainingDataset2DConfig(
            parameters_samples=parameters_samples_K_G,
            edge_length=edge_length,
            radius=radius,
            traction_left=traction_left,
            volume_force=volume_force,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=num_points_per_bc,
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
                    LinearElasticityProblemConfig_K_G(
                        model=material_model,
                        material_parameters=(
                            calculate_K_from_E_and_nu(
                                E=youngs_moduli[i], nu=poissons_ratios[i]
                            ),
                            calculate_G_from_E_and_nu(
                                E=youngs_moduli[i], nu=poissons_ratios[i]
                            ),
                        ),
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
        min_coordinate_x = -edge_length
        max_coordinate_x = 0.0
        min_coordinate_y = 0.0
        max_coordinate_y = edge_length
        min_coordinates = torch.tensor([min_coordinate_x, min_coordinate_y])
        max_coordinates = torch.tensor([max_coordinate_x, max_coordinate_y])

        min_parameters = torch.tensor([min_bulk_modulus, min_shear_modulus])
        max_parameters = torch.tensor([max_bulk_modulus, max_shear_modulus])

        min_inputs = torch.concat((min_coordinates, min_parameters))
        max_inputs = torch.concat((max_coordinates, max_parameters))

        _output_subdir = os.path.join(
            output_subdirectory_preprocessing,
            "results_for_determining_normalization_values",
        )
        print("Run FE simulation to determine normalization values ...")
        domain_config = create_fem_domain_config()
        significant_bulk_modulus = calculate_K_from_E_and_nu(
            E=min_youngs_modulus, nu=max_poissons_ratio
        )
        significant_shear_modulus = calculate_G_from_E_and_nu(
            E=min_youngs_modulus, nu=max_poissons_ratio
        )
        problem_config = LinearElasticityProblemConfig_K_G(
            model=material_model,
            material_parameters=(significant_bulk_modulus, significant_shear_modulus),
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
        material_model=material_model,
        number_points_per_bc=num_points_per_bc,
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
                LinearElasticityProblemConfig_K_G(
                    model=material_model,
                    material_parameters=(
                        calculate_K_from_E_and_nu(
                            E=youngs_moduli[i], nu=poissons_ratios[i]
                        ),
                        calculate_G_from_E_and_nu(
                            E=youngs_moduli[i], nu=poissons_ratios[i]
                        ),
                    ),
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
    exact_youngs_modulus = 195000
    exact_poissons_ratio = 0.35
    exact_bulk_modulus = calculate_K_from_E_and_nu(
        E=exact_youngs_modulus, nu=exact_poissons_ratio
    )
    exact_shear_modulus = calculate_G_from_E_and_nu(
        E=exact_youngs_modulus, nu=exact_poissons_ratio
    )
    num_data_points = 128
    std_noise = 5 * 1e-4

    def generate_calibration_data() -> tuple[Tensor, Tensor]:
        domain_config = create_fem_domain_config()
        problem_config = LinearElasticityProblemConfig_K_G(
            model=material_model,
            material_parameters=(exact_bulk_modulus, exact_shear_modulus),
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
    likelihood = create_standard_ppinn_likelihood_for_noise(
        model=model,
        data=data,
        num_model_parameters=num_material_parameters,
        device=device,
    )

    prior_mean_youngs_modulus = 210000
    prior_mean_poissons_ratio = 0.3
    prior_mean_bulk_modulus = calculate_K_from_E_and_nu(
        E=prior_mean_youngs_modulus, nu=prior_mean_poissons_ratio
    )
    prior_std_bulk_modulus = 10000
    prior_mean_shear_modulus = calculate_G_from_E_and_nu(
        E=prior_mean_youngs_modulus, nu=prior_mean_poissons_ratio
    )
    prior_std_shear_modulus = 1000

    prior_bulk_modulus = create_univariate_normal_distributed_prior(
        mean=prior_mean_bulk_modulus,
        standard_deviation=prior_std_bulk_modulus,
        device=device,
    )
    prior_shear_modulus = create_univariate_normal_distributed_prior(
        mean=prior_mean_shear_modulus,
        standard_deviation=prior_std_shear_modulus,
        device=device,
    )
    prior = multiply_priors([prior_bulk_modulus, prior_shear_modulus])

    parameter_names = ("bulk modulus", "shear modulus")
    true_parameters = (exact_bulk_modulus, exact_shear_modulus)
    initial_parameters = torch.tensor(
        [prior_mean_bulk_modulus, prior_mean_shear_modulus], device=device
    )

    mean_displacements = torch.mean(torch.absolute(noisy_displacements), dim=0)
    residual_weights = 1 / mean_displacements
    print(f"Used residual weights: {residual_weights}")

    least_squares_config = LeastSquaresConfig(
        ansatz=model,
        calibration_data=data,
        initial_parameters=initial_parameters,
        num_iterations=100,
        resdiual_weights=residual_weights.to(device)
        .repeat((num_data_points, 1))
        .ravel(),
    )

    std_proposal_density_bulk_modulus = 100
    std_proposal_density_shear_modulus = 50
    mcmc_config_mh = MetropolisHastingsConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e5),
        num_burn_in_iterations=int(5e4),
        cov_proposal_density=torch.diag(
            torch.tensor(
                [
                    std_proposal_density_bulk_modulus,
                    std_proposal_density_shear_modulus,
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
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e4),
        num_leabfrog_steps=256,
        leapfrog_step_sizes=torch.tensor([1, 1], device=device),
    )
    mcmc_config_enuts = EfficientNUTSConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e4),
        max_tree_depth=8,
        leapfrog_step_sizes=torch.tensor([1, 1], device=device),
    )
    if use_least_squares:
        start = perf_counter()
        identified_parameters, _ = calibrate(
            calibration_config=least_squares_config,
            device=device,
        )
        end = perf_counter()
        time = end - start
        identified_K = identified_parameters[0]
        identified_G = identified_parameters[1]
        identified_E = calculate_E_from_K_and_G(K=identified_K, G=identified_G)
        identified_nu = calculate_nu_from_K_and_G(K=identified_K, G=identified_G)
        print(f"Identified parameters: K = {identified_K} and G = {identified_G}")
        print(f"Identified parameters: E = {identified_E} and nu = {identified_nu}")
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
        print(
            f"Identified moments (normal distribution assumed): {posterior_moments_mh}"
        )
        print(f"Run time Metropolis-Hasting: {time}")
        print("############################################################")
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
        print(
            f"Identified moments (normal distribution assumed): {posterior_moments_h}"
        )
        print(f"Run time Hamiltonian: {time}")
        print("############################################################")
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
        print(
            f"Identified moments (normal distribution assumed): {posterior_moments_enuts}"
        )
        print(f"Run time efficient NUTS: {time}")
        print("############################################################")
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
