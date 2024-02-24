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
from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import (
    create_univariate_normal_distributed_prior,
    multiply_priors,
)
from parametricpinn.calibration import (
    CalibrationData,
    CalibrationDataLoader2D,
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
    NeoHookeProblemConfig,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    generate_validation_data,
    run_simulation,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader, PandasDataWriter
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig2D,
    plot_displacements_2d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_standard_neohooke_quarterplatewithhole import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import NPArray, Tensor

### Configuration
retrain_parametric_pinn = False
# Set up
num_material_parameters = 2
edge_length = 100.0
radius = 10.0
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
mean_bulk_modulus = 6000.0
standard_deviation_bulk_modulus = 400.0
min_bulk_modulus = mean_bulk_modulus - 3 * standard_deviation_bulk_modulus
max_bulk_modulus = mean_bulk_modulus + 3 * standard_deviation_bulk_modulus
mean_shear_modulus = 1000.0
standard_deviation_shear_modulus = 200.0
min_shear_modulus = mean_shear_modulus - 3 * standard_deviation_shear_modulus
max_shear_modulus = mean_shear_modulus + 3 * standard_deviation_shear_modulus
# Network
layer_sizes = [4, 128, 128, 128, 128, 128, 128, 2]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_per_parameter = 32
num_collocation_points = 64
number_points_per_bc = 64
bcs_overlap_distance = 1e-2
bcs_overlap_angle_distance = 1e-2
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 15000
weight_pde_loss = 1.0
weight_stress_bc_loss = 1.0
weight_traction_bc_loss = 1.0
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 2
fem_element_size = 0.2
# Validation
regenerate_valid_data = False
input_subdir_valid = f"20240223_validation_data_neohooke_quarterplatewithhole_K_{int(min_bulk_modulus)}_{int(max_bulk_modulus)}_G_{int(min_shear_modulus)}_{int(max_shear_modulus)}_edge_{int(edge_length)}_radius_{int(radius)}_traction_{int(traction_left_x)}_elementsize_{fem_element_size}"
num_samples_valid = 100
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
consider_model_error = True
use_least_squares = True
use_random_walk_metropolis_hasting = True
use_hamiltonian = False
use_efficient_nuts = False
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = "20240220"
output_subdirectory = f"{output_date}_parametric_pinn_neohooke_quarterplatewithhole_K_{int(min_bulk_modulus)}_{int(max_bulk_modulus)}_G_{int(min_shear_modulus)}_{int(max_shear_modulus)}_col_{int(num_collocation_points)}_bc_{int(number_points_per_bc)}_neurons_6_128"
output_subdirectory_training = os.path.join(output_subdirectory, "training")
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
            min_parameters=[min_bulk_modulus, min_shear_modulus],
            max_parameters=[max_bulk_modulus, max_shear_modulus],
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
            shear_moduli = _generate_random_parameter_list(
                num_samples_valid, min_shear_modulus, max_shear_modulus
            )
            domain_config = create_fem_domain_config()
            problem_configs = []
            for i in range(num_samples_valid):
                problem_configs.append(
                    NeoHookeProblemConfig(
                        material_parameters=(bulk_moduli[i], shear_moduli[i]),
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

            min_parameters = torch.tensor([min_bulk_modulus, min_shear_modulus])
            max_parameters = torch.tensor([max_bulk_modulus, max_shear_modulus])

            min_inputs = torch.concat((min_coordinates, min_parameters))
            max_inputs = torch.concat((max_coordinates, max_parameters))

            print(
                "Run FE simulations to determine normalization values in x-direction ..."
            )
            domain_config = create_fem_domain_config()
            problem_config_x = NeoHookeProblemConfig(
                material_parameters=(min_bulk_modulus, min_shear_modulus)
            )
            simulation_config_x = SimulationConfig(
                domain_config=domain_config,
                problem_config=problem_config_x,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
            )
            output_subdir_x = os.path.join(
                normalization_values_subdir, "displacements_x"
            )
            simulation_results_x = run_simulation(
                simulation_config=simulation_config_x,
                save_results=True,
                save_metadata=True,
                output_subdir=output_subdir_x,
                project_directory=project_directory,
            )

            print(
                "Run FE simulations to determine normalization values in y-direction ..."
            )
            problem_config_y = NeoHookeProblemConfig(
                material_parameters=(max_bulk_modulus, min_shear_modulus)
            )
            simulation_config_y = SimulationConfig(
                domain_config=domain_config,
                problem_config=problem_config_y,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
            )
            output_subdir_y = os.path.join(
                normalization_values_subdir, "displacements_y"
            )
            simulation_results_y = run_simulation(
                simulation_config=simulation_config_y,
                save_results=True,
                save_metadata=True,
                output_subdir=output_subdir_y,
                project_directory=project_directory,
            )

            min_displacement_x = float(np.amin(simulation_results_x.displacements_x))
            max_displacement_x = float(np.amax(simulation_results_x.displacements_x))
            min_displacement_y = float(np.amin(simulation_results_y.displacements_y))
            max_displacement_y = float(np.amax(simulation_results_y.displacements_y))
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
        output_subdirectory=output_subdirectory_training,
        project_directory=project_directory,
        device=device,
    )

    def _plot_exemplary_displacement_fields() -> None:
        displacements_plotter_config = DisplacementsPlotterConfig2D()
        parameters_list = [
            (min_bulk_modulus, min_shear_modulus),
            (min_bulk_modulus, max_shear_modulus),
            (max_bulk_modulus, min_shear_modulus),
            (max_bulk_modulus, max_shear_modulus),
        ]
        bulk_moduli, shear_moduli = zip(*parameters_list)

        domain_config = create_fem_domain_config()
        problem_configs = []
        for i in range(len(parameters_list)):
            problem_configs.append(
                NeoHookeProblemConfig(
                    material_parameters=(bulk_moduli[i], shear_moduli[i])
                )
            )

        plot_displacements_2d(
            ansatz=ansatz,
            domain_config=domain_config,
            problem_configs=problem_configs,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            output_subdir=output_subdirectory_training,
            project_directory=project_directory,
            plot_config=displacements_plotter_config,
            device=device,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacement_fields()


def calibration_step() -> None:
    print("Start calibration ...")
    num_data_points = 1024
    std_noise = 5 * 1e-4
    num_test_cases = num_samples_valid
    prior_mean_bulk_modulus = mean_bulk_modulus
    prior_std_bulk_modulus = standard_deviation_bulk_modulus
    prior_mean_shear_modulus = mean_shear_modulus
    prior_std_shear_modulus = standard_deviation_shear_modulus

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
    initial_parameters = torch.tensor(
        [prior_mean_bulk_modulus, prior_mean_shear_modulus], device=device
    )

    def generate_calibration_data() -> tuple[tuple[CalibrationData, ...], NPArray]:
        calibration_data_loader = CalibrationDataLoader2D(
            input_subdir=input_subdir_valid,
            num_data_points=num_data_points,
            std_noise=std_noise,
            num_cases=num_test_cases,
            project_directory=project_directory,
            device=device,
        )
        calibration_data_sets, true_parameters = calibration_data_loader.get_data()
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
            std_proposal_density_bulk_modulus = 5.0
            std_proposal_density_shear_modulus = 1.0
            cov_proposal_density = torch.diag(
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
                num_burn_in_iterations=int(1e4),
                num_leabfrog_steps=256,
                leapfrog_step_sizes=torch.tensor([1, 0.01], device=device),
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
                num_burn_in_iterations=int(1e4),
                max_tree_depth=8,
                leapfrog_step_sizes=torch.tensor([1, 0.01], device=device),
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


training_dataset, validation_dataset = create_datasets()
if retrain_parametric_pinn:
    training_step()
calibration_step()
