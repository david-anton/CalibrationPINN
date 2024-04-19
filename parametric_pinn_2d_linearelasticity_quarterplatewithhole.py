import os
from datetime import date
from time import perf_counter
from typing import TypeAlias

import numpy as np
import pandas as pd
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_quarter_plate_with_hole,
)
from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import (
    create_gamma_distributed_prior,
    create_univariate_uniform_distributed_prior,
    multiply_priors,
)
from parametricpinn.calibration import (
    CalibrationData,
    CalibrationDataLoader2D,
    EfficientNUTSConfig,
    EMCEEConfig,
    HamiltonianConfig,
    LeastSquaresConfig,
    MetropolisHastingsConfig,
    test_coverage,
    test_least_squares_calibration,
)
from parametricpinn.calibration.bayesianinference.likelihoods import (
    create_optimized_standard_ppinn_likelihood_for_noise_and_model_error_gps,
    create_standard_ppinn_likelihood_for_noise,
    create_standard_ppinn_likelihood_for_noise_and_model_error_gps_sampling,
    create_standard_ppinn_q_likelihood_for_noise,
)
from parametricpinn.calibration.data import concatenate_calibration_data
from parametricpinn.calibration.utility import load_model
from parametricpinn.data.parameterssampling import sample_quasirandom_sobol
from parametricpinn.data.simulation_2d import (
    SimulationDataset2D,
    SimulationDataset2DConfig,
    create_simulation_dataset,
)
from parametricpinn.data.trainingdata_2d import (
    QuarterPlateWithHoleTrainingDataset2D,
    QuarterPlateWithHoleTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.errors import UnvalidMainConfigError
from parametricpinn.fem import (
    LinearElasticityProblemConfig_K_G,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    generate_simulation_data,
    run_simulation,
)
from parametricpinn.gps import IndependentMultiOutputGP, create_gaussian_process
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader, PandasDataWriter
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig2D,
    plot_displacements_2d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.loss_2d.momentum_linearelasticity_K_G import (
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
)
from parametricpinn.training.training_standard_linearelasticity_quarterplatewithhole import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import NPArray, Tensor

### Configuration
retrain_parametric_pinn = False
# Set up
material_model = "plane stress"
num_material_parameters = 2
edge_length = 100.0
radius = 10.0
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_bulk_modulus = 100000.0
max_bulk_modulus = 200000.0
min_shear_modulus = 60000.0
max_shear_modulus = 100000.0
# Network
layer_sizes = [4, 128, 128, 128, 128, 128, 128, 2]
activation = torch.nn.Tanh()
# Ansatz
distance_function = "normalized linear"
# Training
num_parameter_samples_pinn = 1024
num_collocation_points = 64
num_points_per_bc = 64
bcs_overlap_distance = 1e-2
bcs_overlap_angle_distance = 1e-2
training_batch_size = num_parameter_samples_pinn
use_simulation_data = True
regenerate_train_data = False
num_parameter_samples_data = 128
num_data_points = 128
number_training_epochs = 10000
weight_pde_loss = 1.0
weight_stress_bc_loss = 1.0
weight_traction_bc_loss = 1.0
weight_data_loss = 1e6
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size = 0.1
# Validation
regenerate_valid_data = False
input_subdir_valid = f"20240410_validation_data_linearelasticity_quarterplatewithhole_K_{min_bulk_modulus}_{max_bulk_modulus}_G_{min_shear_modulus}_{max_shear_modulus}_edge_{int(edge_length)}_radius_{int(radius)}_traction_{int(traction_left_x)}_elementsize_{fem_element_size}"
num_samples_valid = 100
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
calibration_method = "noise_only"
# calibration_method = "noise_and_q_likelihood"
# calibration_method = "overestimated_error_stds"
# calibration_method = "full_bayes_with_error_gps"
# calibration_method = "empirical_bayes_with_error_gps"
use_least_squares = True
use_random_walk_metropolis_hasting = False
use_emcee = True
use_hamiltonian = False
use_efficient_nuts = False
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = "20240414"
output_subdirectory = f"{output_date}_parametric_pinn_linearelasticity_quarterplatewithhole_K_{min_bulk_modulus}_{max_bulk_modulus}_G_{min_shear_modulus}_{max_shear_modulus}_pinnsamples_{num_parameter_samples_pinn}_col_{num_collocation_points}_bc_{num_points_per_bc}_datasamples_{num_parameter_samples_data}_neurons_6_128"
output_subdir_training = os.path.join(output_subdirectory, "training")
output_subdir_normalization = os.path.join(output_subdir_training, "normalization")
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


# Convert material parameters
calculate_K_from_E_and_nu = calculate_K_from_E_and_nu_factory(material_model)


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
    tuple[
        QuarterPlateWithHoleTrainingDataset2D,
        SimulationDataset2D | None,
        SimulationDataset2D,
    ]
):
    def _create_pinn_training_dataset() -> QuarterPlateWithHoleTrainingDataset2D:
        print("Generate training data ...")
        parameters_samples = sample_quasirandom_sobol(
            min_parameters=[min_bulk_modulus, min_shear_modulus],
            max_parameters=[max_bulk_modulus, max_shear_modulus],
            num_samples=num_parameter_samples_pinn,
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
            num_points_per_bc=num_points_per_bc,
            bcs_overlap_distance=bcs_overlap_distance,
            bcs_overlap_angle_distance=bcs_overlap_angle_distance,
        )
        return create_training_dataset(config_training_data)

    def _create_data_training_dataset() -> SimulationDataset2D:
        training_data_subdir = os.path.join(output_subdir_training, "training_data")

        def _generate_data() -> None:
            parameters_samples = sample_quasirandom_sobol(
                min_parameters=[min_bulk_modulus, min_shear_modulus],
                max_parameters=[max_bulk_modulus, max_shear_modulus],
                num_samples=num_parameter_samples_data,
                device=device,
            )
            domain_config = create_fem_domain_config()
            problem_configs = [
                LinearElasticityProblemConfig_K_G(
                    model=material_model,
                    material_parameters=(
                        parameters_sample[0].item(),
                        parameters_sample[1].item(),
                    ),
                )
                for parameters_sample in parameters_samples
            ]

            generate_simulation_data(
                domain_config=domain_config,
                problem_configs=problem_configs,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
                save_metadata=save_metadata,
                output_subdir=training_data_subdir,
                project_directory=project_directory,
                save_to_input_dir=False,
            )

        if regenerate_train_data:
            print("Run FE simulations to generate training data ...")
            _generate_data()
        print("Load training data ...")
        config_validation_data = SimulationDataset2DConfig(
            input_subdir=training_data_subdir,
            num_points=num_data_points,
            num_samples=num_parameter_samples_data,
            project_directory=project_directory,
            read_from_output_dir=True,
        )
        return create_simulation_dataset(config_validation_data)

    def _create_validation_dataset() -> SimulationDataset2D:
        def _generate_validation_data() -> None:
            offset_training_range_bulk_modulus = 1000.0
            offset_training_range_shear_modulus = 500.0

            def _generate_random_parameter_list(
                size: int, min_value: float, max_value: float
            ) -> list[float]:
                random_params = min_value + torch.rand(size) * (max_value - min_value)
                return random_params.tolist()

            bulk_moduli_list = _generate_random_parameter_list(
                num_samples_valid,
                min_bulk_modulus + offset_training_range_bulk_modulus,
                max_bulk_modulus - offset_training_range_bulk_modulus,
            )
            shear_moduli_list = _generate_random_parameter_list(
                num_samples_valid,
                min_shear_modulus + offset_training_range_shear_modulus,
                max_shear_modulus - offset_training_range_shear_modulus,
            )
            domain_config = create_fem_domain_config()
            problem_configs = [
                LinearElasticityProblemConfig_K_G(
                    model=material_model,
                    material_parameters=(bulk_modulus, shear_modulus),
                )
                for bulk_modulus, shear_modulus in zip(
                    bulk_moduli_list, shear_moduli_list
                )
            ]
            generate_simulation_data(
                domain_config=domain_config,
                problem_configs=problem_configs,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
                save_metadata=save_metadata,
                output_subdir=input_subdir_valid,
                project_directory=project_directory,
            )

        if regenerate_valid_data:
            print("Run FE simulations to generate validation data ...")
            _generate_validation_data()
        else:
            print("Load validation data ...")
        config_validation_data = SimulationDataset2DConfig(
            input_subdir=input_subdir_valid,
            num_points=num_points_valid,
            num_samples=num_samples_valid,
            project_directory=project_directory,
        )
        return create_simulation_dataset(config_validation_data)

    training_dataset_pinn = _create_pinn_training_dataset()
    if use_simulation_data:
        training_dataset_data = _create_data_training_dataset()
    else:
        training_dataset_data = None
    validation_dataset = _create_validation_dataset()
    return training_dataset_pinn, training_dataset_data, validation_dataset


def create_ansatz() -> StandardAnsatz:
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
                subdir_name=output_subdir_normalization,
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
                subdir_name=output_subdir_normalization,
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

            min_parameters = torch.tensor([min_bulk_modulus, min_shear_modulus])
            max_parameters = torch.tensor([max_bulk_modulus, max_shear_modulus])

            min_inputs = torch.concat((min_coordinates, min_parameters))
            max_inputs = torch.concat((max_coordinates, max_parameters))

            print(
                "Run FE simulations to determine normalization values in x-direction ..."
            )
            domain_config = create_fem_domain_config()
            problem_config_x = LinearElasticityProblemConfig_K_G(
                model=material_model,
                material_parameters=(min_bulk_modulus, min_shear_modulus),
            )
            simulation_config_x = SimulationConfig(
                domain_config=domain_config,
                problem_config=problem_config_x,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
            )
            results_output_subdir_x = os.path.join(
                output_subdir_normalization, "fem_simulation_results_displacements_x"
            )
            simulation_results_x = run_simulation(
                simulation_config=simulation_config_x,
                save_results=True,
                save_metadata=True,
                output_subdir=results_output_subdir_x,
                project_directory=project_directory,
            )

            print(
                "Run FE simulations to determine normalization values in y-direction ..."
            )
            problem_config_y = LinearElasticityProblemConfig_K_G(
                model=material_model,
                material_parameters=(max_bulk_modulus, min_shear_modulus),
            )
            simulation_config_y = SimulationConfig(
                domain_config=domain_config,
                problem_config=problem_config_y,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
            )
            results_output_subdir_y = os.path.join(
                output_subdir_normalization, "fem_simulation_results_displacements_y"
            )
            simulation_results_y = run_simulation(
                simulation_config=simulation_config_y,
                save_results=True,
                save_metadata=True,
                output_subdir=results_output_subdir_y,
                project_directory=project_directory,
            )

            min_displacement_x = float(np.amin(simulation_results_x.displacements_x))
            max_displacement_x = float(np.amax(simulation_results_x.displacements_x))
            min_displacement_y = float(np.amin(simulation_results_y.displacements_y))
            max_displacement_y = float(np.amax(simulation_results_y.displacements_y))
            min_outputs = torch.tensor([min_displacement_x, min_displacement_y])
            max_outputs = torch.tensor([max_displacement_x, max_displacement_y])
            normalization_values = {
                key_min_inputs: min_inputs.to(device),
                key_max_inputs: max_inputs.to(device),
                key_min_outputs: min_outputs.to(device),
                key_max_outputs: max_outputs.to(device),
            }
            _save_normalization_values(normalization_values)
        else:
            normalization_values = _read_normalization_values()
        _print_normalization_values(normalization_values)
        return normalization_values

    normalization_values = _determine_normalization_values()
    network = FFNN(layer_sizes=layer_sizes, activation=activation)
    return create_standard_normalized_hbc_ansatz_quarter_plate_with_hole(
        min_inputs=normalization_values[key_min_inputs],
        max_inputs=normalization_values[key_max_inputs],
        min_outputs=normalization_values[key_min_outputs],
        max_outputs=normalization_values[key_max_outputs],
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
        weight_data_loss=weight_data_loss,
        training_dataset_pinn=training_dataset_pinn,
        number_training_epochs=number_training_epochs,
        training_batch_size=training_batch_size,
        validation_dataset=validation_dataset,
        validation_interval=validation_interval,
        output_subdirectory=output_subdir_training,
        project_directory=project_directory,
        device=device,
        training_dataset_data=training_dataset_data,
    )

    def _plot_exemplary_displacement_fields() -> None:
        displacements_plotter_config = DisplacementsPlotterConfig2D()
        material_parameters_list = [
            (min_bulk_modulus, min_shear_modulus),
            (min_bulk_modulus, max_shear_modulus),
            (max_bulk_modulus, min_shear_modulus),
            (max_bulk_modulus, max_shear_modulus),
            (
                calculate_K_from_E_and_nu(E=210000.0, nu=0.3),
                calculate_G_from_E_and_nu(E=210000.0, nu=0.3),
            ),
        ]

        domain_config = create_fem_domain_config()
        problem_configs = [
            LinearElasticityProblemConfig_K_G(
                model=material_model, material_parameters=(bulk_modulus, shear_modulus)
            )
            for bulk_modulus, shear_modulus in material_parameters_list
        ]

        plot_displacements_2d(
            ansatz=ansatz,
            domain_config=domain_config,
            problem_configs=problem_configs,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            output_subdir=output_subdir_training,
            project_directory=project_directory,
            plot_config=displacements_plotter_config,
            device=device,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacement_fields()


def calibration_step() -> None:
    print("Start calibration ...")
    num_test_cases = num_samples_valid
    num_data_sets = 1
    num_data_points = 128
    std_noise = 5 * 1e-4

    material_parameter_names = ("bulk modulus", "shear modulus")

    initial_bulk_modulus = 160000.0
    initial_shear_modulus = 79000.0
    initial_material_parameters = torch.tensor(
        [initial_bulk_modulus, initial_shear_modulus],
        device=device,
    )

    prior_bulk_modulus = create_univariate_uniform_distributed_prior(
        lower_limit=min_bulk_modulus, upper_limit=max_bulk_modulus, device=device
    )
    prior_shear_modulus = create_univariate_uniform_distributed_prior(
        lower_limit=min_shear_modulus, upper_limit=max_shear_modulus, device=device
    )
    prior_material_parameters = multiply_priors(
        [prior_bulk_modulus, prior_shear_modulus]
    )

    def generate_calibration_data() -> tuple[tuple[CalibrationData, ...], NPArray]:
        calibration_data_loader = CalibrationDataLoader2D(
            input_subdir=input_subdir_valid,
            num_cases=num_test_cases,
            num_data_sets=num_data_sets,
            num_data_points=num_data_points,
            std_noise=std_noise,
            project_directory=project_directory,
            device=device,
        )
        calibration_data, true_parameters = calibration_data_loader.load_data()
        return calibration_data, true_parameters.detach().cpu().numpy()

    name_model_parameters_file = "model_parameters"
    model = load_model(
        model=ansatz,
        name_model_parameters_file=name_model_parameters_file,
        input_subdir=output_subdir_training,
        project_directory=project_directory,
        device=device,
    )

    calibration_data, true_material_parameters = generate_calibration_data()

    def create_model_error_gp() -> IndependentMultiOutputGP:
        min_inputs = torch.tensor(
            [-edge_length, 0.0],
            dtype=torch.float64,
            device=device,
        )
        max_inputs = torch.tensor(
            [0.0, edge_length],
            dtype=torch.float64,
            device=device,
        )
        return IndependentMultiOutputGP(
            gps=[
                create_gaussian_process(
                    mean="zero",
                    kernel="scaled_rbf",
                    min_inputs=min_inputs,
                    max_inputs=max_inputs,
                    device=device,
                ),
                create_gaussian_process(
                    mean="zero",
                    kernel="scaled_rbf",
                    min_inputs=min_inputs,
                    max_inputs=max_inputs,
                    device=device,
                ),
            ],
            device=device,
        ).to(device)

    output_subdir_calibration = os.path.join(
        output_subdirectory, "calibration", calibration_method
    )
    output_subdir_likelihoods = os.path.join(output_subdir_calibration, "likelihoods")

    model_error_gp = create_model_error_gp()

    gp_parameter_names = (
        "output_scale_0",
        "length_scale_0",
        "output_scale_1",
        "length_scale_1",
    )
    initial_gp_output_scale = 1e-2
    initial_gp_length_scale = 1e-2
    initial_model_error_gp_parameters = torch.tensor(
        [
            initial_gp_output_scale,
            initial_gp_length_scale,
            initial_gp_output_scale,
            initial_gp_length_scale,
        ],
        dtype=torch.float64,
        device=device,
    )

    ParameterNames: TypeAlias = tuple[str, str] | tuple[str, str, str, str, str, str]

    if calibration_method == "noise_only":
        likelihoods = tuple(
            create_standard_ppinn_likelihood_for_noise(
                model=model,
                num_model_parameters=num_material_parameters,
                data=data,
                device=device,
            )
            for _, data in enumerate(calibration_data)
        )

        prior = prior_material_parameters
        parameter_names: ParameterNames = material_parameter_names
        initial_parameters = initial_material_parameters

        std_proposal_density_bulk_modulus = 2000.0
        std_proposal_density_shear_modulus = 100.0
        covar_rwmh_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_bulk_modulus,
                    std_proposal_density_shear_modulus,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )
        num_rwmh_iterations = int(2e4)
        num_rwmh_burn_in_iterations = int(1e4)

    elif calibration_method == "noise_and_q_likelihood":
        likelihoods = tuple(
            create_standard_ppinn_q_likelihood_for_noise(
                model=model,
                num_model_parameters=num_material_parameters,
                data=data,
                device=device,
            )
            for _, data in enumerate(calibration_data)
        )

        prior = prior_material_parameters
        parameter_names = material_parameter_names
        initial_parameters = initial_material_parameters

        std_proposal_density_bulk_modulus = 2000.0
        std_proposal_density_shear_modulus = 100.0
        covar_rwmh_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_bulk_modulus,
                    std_proposal_density_shear_modulus,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )
        num_rwmh_iterations = int(2e4)
        num_rwmh_burn_in_iterations = int(1e4)

    elif calibration_method == "overestimated_error_stds":
        std_model_error = 1e-3
        std_noise_and_model_error = std_noise + std_model_error

        for data in calibration_data:
            data.std_noise = std_noise_and_model_error

        likelihoods = tuple(
            create_standard_ppinn_likelihood_for_noise(
                model=model,
                num_model_parameters=num_material_parameters,
                data=data,
                device=device,
            )
            for _, data in enumerate(calibration_data)
        )

        prior = prior_material_parameters
        parameter_names = material_parameter_names
        initial_parameters = initial_material_parameters

        std_proposal_density_bulk_modulus = 2000.0
        std_proposal_density_shear_modulus = 100.0
        covar_rwmh_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_bulk_modulus,
                    std_proposal_density_shear_modulus,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )
        num_rwmh_iterations = int(2e4)
        num_rwmh_burn_in_iterations = int(1e4)

    elif calibration_method == "full_bayes_with_error_gps":
        prior_output_scale = create_gamma_distributed_prior(
            concentration=1.01, rate=10.0, device=device
        )
        prior_length_scale = create_gamma_distributed_prior(
            concentration=1.01, rate=10.0, device=device
        )
        prior_gp_parameters = multiply_priors(
            [
                prior_output_scale,
                prior_length_scale,
                prior_output_scale,
                prior_length_scale,
            ]
        )

        likelihoods = tuple(
            create_standard_ppinn_likelihood_for_noise_and_model_error_gps_sampling(
                model=model,
                num_model_parameters=num_material_parameters,
                model_error_gp=model_error_gp,
                data=data,
                device=device,
            )
            for _, data in enumerate(calibration_data)
        )

        prior = multiply_priors([prior_material_parameters, prior_gp_parameters])
        parameter_names = material_parameter_names + gp_parameter_names
        initial_parameters = torch.concat(
            (initial_material_parameters, initial_model_error_gp_parameters)
        )

        std_proposal_density_bulk_modulus = 200.0
        std_proposal_density_shear_modulus = 50.0
        std_gp_output_scale = 1e-5
        std_gp_length_scale = 1e-4
        covar_rwmh_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_bulk_modulus,
                    std_proposal_density_shear_modulus,
                    std_gp_output_scale,
                    std_gp_length_scale,
                    std_gp_output_scale,
                    std_gp_length_scale,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )
        num_rwmh_iterations = int(1e5)
        num_rwmh_burn_in_iterations = int(2e5)

    elif calibration_method == "empirical_bayes_with_error_gps":
        model_error_optimization_num_material_parameter_samples = 256
        model_error_optimization_num_iterations = 16

        likelihoods = tuple(
            create_optimized_standard_ppinn_likelihood_for_noise_and_model_error_gps(
                model=model,
                num_model_parameters=num_material_parameters,
                model_error_gp=model_error_gp,
                initial_model_error_gp_parameters=initial_model_error_gp_parameters,
                use_independent_model_error_gps=True,
                data=data,
                prior_material_parameters=prior_material_parameters,
                num_material_parameter_samples=model_error_optimization_num_material_parameter_samples,
                num_iterations=model_error_optimization_num_iterations,
                test_case_index=test_case_index,
                output_subdirectory=output_subdir_likelihoods,
                project_directory=project_directory,
                device=device,
            )
            for test_case_index, data in enumerate(calibration_data)
        )

        prior = prior_material_parameters
        parameter_names = material_parameter_names
        initial_parameters = initial_material_parameters

        std_proposal_density_bulk_modulus = 2000.0
        std_proposal_density_shear_modulus = 100.0
        covar_rwmh_proposal_density = torch.diag(
            torch.tensor(
                [
                    std_proposal_density_bulk_modulus,
                    std_proposal_density_shear_modulus,
                ],
                dtype=torch.float64,
                device=device,
            )
            ** 2
        )
        num_rwmh_iterations = int(2e4)
        num_rwmh_burn_in_iterations = int(1e4)

    else:
        raise UnvalidMainConfigError(
            f"There is no implementation for the requested method: {calibration_method}"
        )

    def set_up_least_squares_configs(
        calibration_data: tuple[CalibrationData, ...],
    ) -> tuple[LeastSquaresConfig, ...]:
        configs = []
        for data in calibration_data:
            concatenated_data = concatenate_calibration_data(data)
            mean_displacements = torch.mean(
                torch.absolute(concatenated_data.outputs), dim=0
            )
            residual_weights = (
                (1 / mean_displacements)
                .to(device)
                .repeat((concatenated_data.num_data_points, 1))
            )
            config = LeastSquaresConfig(
                ansatz=model,
                calibration_data=concatenated_data,
                initial_parameters=initial_material_parameters,
                num_iterations=1000,
                resdiual_weights=residual_weights,
            )
            configs.append(config)
        return tuple(configs)

    def set_up_metropolis_hastings_configs(
        likelihoods: tuple[Likelihood, ...],
    ) -> tuple[MetropolisHastingsConfig, ...]:
        configs = []
        for likelihood in likelihoods:
            config = MetropolisHastingsConfig(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=initial_parameters,
                num_iterations=num_rwmh_iterations,
                num_burn_in_iterations=num_rwmh_burn_in_iterations,
                cov_proposal_density=covar_rwmh_proposal_density,
            )
            configs.append(config)
        return tuple(configs)

    def set_up_emcee_config(
        likelihoods: tuple[Likelihood, ...]
    ) -> tuple[EMCEEConfig, ...]:
        num_walkers = 100
        min_material_parameters = torch.tensor([min_bulk_modulus, min_shear_modulus])
        max_material_parameters = torch.tensor([max_bulk_modulus, max_shear_modulus])
        range_material_parameters = max_material_parameters - min_material_parameters
        initial_parameters = min_material_parameters + torch.rand(
            (num_walkers, num_material_parameters)
        ) * range_material_parameters.repeat((num_walkers, 1))
        configs = []
        for likelihood in likelihoods:
            config = EMCEEConfig(
                likelihood=likelihood,
                prior=prior,
                initial_parameters=initial_parameters.to(device),
                stretch_scale=4.0,
                num_walkers=num_walkers,
                num_iterations=200,
                num_burn_in_iterations=100,
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
                num_burn_in_iterations=int(5e3),
                num_leabfrog_steps=128,
                leapfrog_step_sizes=torch.tensor([1, 1], device=device),
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
                num_burn_in_iterations=int(5e3),
                max_tree_depth=7,
                leapfrog_step_sizes=torch.tensor([1, 1], device=device),
            )
            configs.append(config)
        return tuple(configs)

    if use_least_squares:
        configs_ls = set_up_least_squares_configs(calibration_data)
        start = perf_counter()
        test_least_squares_calibration(
            calibration_configs=configs_ls,
            parameter_names=material_parameter_names,
            true_parameters=true_material_parameters,
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
            true_parameters=true_material_parameters,
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
    if use_emcee:
        configs_emcee = set_up_emcee_config(likelihoods)
        start = perf_counter()
        test_coverage(
            calibration_configs=configs_emcee,
            parameter_names=parameter_names,
            true_parameters=true_material_parameters,
            output_subdir=os.path.join(output_subdir_calibration, "emcee"),
            project_directory=project_directory,
            device=device,
        )
        end = perf_counter()
        time = end - start
        print(f"Run time emcee coverage test: {time}")
        print("############################################################")
    if use_hamiltonian:
        configs_h = set_up_hamiltonian_configs(likelihoods)
        start = perf_counter()
        test_coverage(
            calibration_configs=configs_h,
            parameter_names=parameter_names,
            true_parameters=true_material_parameters,
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
            true_parameters=true_material_parameters,
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
    training_dataset_pinn, training_dataset_data, validation_dataset = create_datasets()
    training_step()
calibration_step()
