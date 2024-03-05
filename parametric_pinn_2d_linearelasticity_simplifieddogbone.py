import os
from datetime import date
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_clamped_left,
)
from parametricpinn.bayesian.likelihood import Likelihood
from parametricpinn.bayesian.prior import (
    create_univariate_uniform_distributed_prior,
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
    SimplifiedDogBoneGeometryConfig,
    SimplifiedDogBoneTrainingDataset2D,
    SimplifiedDogBoneTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.data.validationdata_2d import (
    ValidationDataset2D,
    ValidationDataset2DConfig,
    create_validation_dataset,
)
from parametricpinn.fem import (
    LinearElasticityProblemConfig_K_G,
    SimplifiedDogBoneDomainConfig,
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
from parametricpinn.training.loss_2d.momentum_linearelasticity_K_G import (
    calculate_E_from_K_and_G_factory,
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
    calculate_nu_from_K_and_G_factory,
)
from parametricpinn.training.training_standard_linearelasticity_simplifieddogbone import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import NPArray, Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
material_model = "plane stress"
num_material_parameters = 2
traction_right_x = 106.2629  # [N/mm^2]
traction_right_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 160000.0
max_youngs_modulus = 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = 0.4
# Network
layer_sizes = [4, 128, 128, 128, 128, 128, 128, 2]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_per_parameter = 32
num_collocation_points = 64
num_points_per_bc = 64
bcs_overlap_angle_distance_left = 1e-2
bcs_overlap_distance_parallel_right = 1e-2
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 30000
weight_pde_loss = 1.0
weight_traction_bc_loss = 1.0
weight_symmetry_bc_loss = 1e5
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size = 0.1
# Validation
regenerate_valid_data = True
input_subdir_valid = "20240304_validation_data_linearelasticity_simplifieddogbone_E_160k_240k_nu_02_04_elementsize_01_K_G"
num_samples_valid = 100
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
input_subdir_calibration = os.path.join(
    "Paper_PINNs", "20240123_experimental_dic_data_dogbone"
)
input_file_name_calibration = "displacements_dic.csv"
consider_model_error = True
use_least_squares = True
use_random_walk_metropolis_hasting = True
use_hamiltonian = False
use_efficient_nuts = False
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_parametric_pinn_linearelasticity_simplifieddogbone_E_{int(min_youngs_modulus)}_{int(max_youngs_modulus)}_nu_{int(min_poissons_ratio)}_{int(max_poissons_ratio)}_samples_{num_samples_per_parameter}_col_{num_collocation_points}_bc_{num_points_per_bc}_neurons_6_128"
output_subdirectory_training = os.path.join(output_subdirectory, "training")
output_subdirectory_preprocessing = os.path.join(output_subdirectory, "preprocessing")
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

# Convert material parameters
calculate_K_from_E_and_nu = calculate_K_from_E_and_nu_factory(material_model)
calculate_E_from_K_and_G = calculate_E_from_K_and_G_factory(material_model)
calculate_nu_from_K_and_G = calculate_nu_from_K_and_G_factory(material_model)
if material_model == "plane stress":
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
if material_model == "plane strain":
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
        parameters_samples = sample_uniform_grid(
            min_parameters=[min_bulk_modulus, min_shear_modulus],
            max_parameters=[max_bulk_modulus, max_shear_modulus],
            num_steps=[num_samples_per_parameter, num_samples_per_parameter],
            device=device,
        )

        traction_right = torch.tensor([traction_right_x, traction_right_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        config_training_data = SimplifiedDogBoneTrainingDataset2DConfig(
            parameters_samples=parameters_samples,
            traction_right=traction_right,
            volume_force=volume_force,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=num_points_per_bc,
            bcs_overlap_angle_distance_left=bcs_overlap_angle_distance_left,
            bcs_overlap_distance_parallel_right=bcs_overlap_distance_parallel_right,
        )
        return create_training_dataset(config_training_data)

    def _create_validation_dataset() -> ValidationDataset2D:
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
            generate_validation_data(
                domain_config=domain_config,
                problem_configs=problem_configs,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
                save_metadata=save_metadata,
                output_subdir=input_subdir_valid,
                project_directory=project_directory,
            )

        if regenerate_valid_data:
            print("Generate validation data ...")
            _generate_validation_data()
        else:
            print("Load validation data ...")
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
            min_coordinate_x = -geometry_config.left_half_box_length
            max_coordinate_x = geometry_config.right_half_box_length
            min_coordinate_y = -geometry_config.half_box_height
            max_coordinate_y = geometry_config.half_box_height
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
            significant_bulk_modulus = calculate_K_from_E_and_nu(
                E=min_youngs_modulus, nu=max_poissons_ratio
            )
            significant_shear_modulus = calculate_G_from_E_and_nu(
                E=min_youngs_modulus, nu=max_poissons_ratio
            )
            problem_config = LinearElasticityProblemConfig_K_G(
                model=material_model,
                material_parameters=(
                    significant_bulk_modulus,
                    significant_shear_modulus,
                ),
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
    return create_standard_normalized_hbc_ansatz_clamped_left(
        coordinate_x_left=torch.tensor(
            [-geometry_config.left_half_box_length], device=device
        ),
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
        weight_pde_loss=weight_pde_loss,
        weight_traction_bc_loss=weight_traction_bc_loss,
        weight_symmetry_bc_loss=weight_symmetry_bc_loss,
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
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            plot_config=displacements_plotter_config,
            device=device,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacement_fields()


def calibration_step() -> None:
    print("Start calibration ...")
    num_data_points = 5240
    std_noise = 5 * 1e-4

    exact_youngs_modulus = 192800.0
    exact_poissons_ratio = 0.2491
    exact_bulk_modulus = calculate_K_from_E_and_nu(
        E=exact_youngs_modulus, nu=exact_poissons_ratio
    )
    exact_shear_modulus = calculate_G_from_E_and_nu(
        E=exact_youngs_modulus, nu=exact_poissons_ratio
    )
    true_parameters = np.array([exact_bulk_modulus, exact_shear_modulus])

    initial_bulk_modulus = 160000.0
    initial_shear_modulus = 79000.0

    prior_bulk_modulus = create_univariate_uniform_distributed_prior(
        lower_limit=min_bulk_modulus, upper_limit=max_bulk_modulus, device=device
    )
    prior_shear_modulus = create_univariate_uniform_distributed_prior(
        lower_limit=min_shear_modulus, upper_limit=max_shear_modulus, device=device
    )

    prior = multiply_priors([prior_bulk_modulus, prior_shear_modulus])
    parameter_names = ("bulk modulus", "shear modulus")
    initial_parameters = torch.tensor(
        [initial_bulk_modulus, initial_shear_modulus], device=device
    )

    def generate_calibration_data() -> CalibrationData:
        csv_reader = CSVDataReader(project_directory)
        data = csv_reader.read(
            file_name=input_file_name_calibration, subdir_name=input_subdir_calibration
        )
        slice_coordinates = slice(0, 2)
        slice_displacements = slice(2, 4)
        full_raw_coordinates = torch.from_numpy(data[:, slice_coordinates]).type(
            torch.float64
        )
        full_raw_displacements = torch.from_numpy(data[:, slice_displacements]).type(
            torch.float64
        )
        # Transform coordinates to the coordinates system used for PINN training
        coordinate_shift_x = geometry_config.left_half_measurement_length
        coordinate_shift_y = geometry_config.half_measurement_height
        full_raw_coordinates = full_raw_coordinates - torch.tensor(
            [coordinate_shift_x, coordinate_shift_y], dtype=torch.float64
        )
        # Filter measurement points within the measurement area
        full_raw_coordinates_x = full_raw_coordinates[:, 0]
        full_raw_coordinates_y = full_raw_coordinates[:, 1]
        left_half_measurement_length = geometry_config.left_half_measurement_length
        right_half_measurement_length = geometry_config.right_half_measurement_length
        half_measurement_height = geometry_config.half_measurement_height
        mask_condition_x = torch.logical_and(
            full_raw_coordinates_x >= -left_half_measurement_length,
            full_raw_coordinates_x <= right_half_measurement_length,
        )

        mask_condition_y = torch.logical_and(
            full_raw_coordinates_y >= -half_measurement_height,
            full_raw_coordinates_y <= half_measurement_height,
        )
        mask_condition = torch.logical_and(mask_condition_x, mask_condition_y)
        mask = torch.where(mask_condition, True, False)
        full_coordinates = full_raw_coordinates[mask]
        full_displacements = full_raw_displacements[mask]
        # Select points for calibration
        size_full_data = len(full_coordinates)
        print(f"Total number of measurement points: {size_full_data}")
        random_indices = torch.randperm(size_full_data)[:num_data_points]
        coordinates = full_coordinates[random_indices, :].to(device)
        displacements = full_displacements[random_indices, :].to(device)

        return CalibrationData(
            inputs=coordinates,
            outputs=displacements,
            std_noise=std_noise,
        )

    def visualize_data(calibration_data: CalibrationData) -> None:
        class PlotterConfigData:
            def __init__(self) -> None:
                # label size
                self.label_size = 16
                # font size in legend
                self.font_size = 16
                self.font = {"size": self.label_size}
                # title pad
                self.title_pad = 10
                # labels
                self.x_label = "x [mm]"
                self.y_label = "y [mm]"
                # major ticks
                self.major_tick_label_size = 16
                self.major_ticks_size = self.font_size
                self.major_ticks_width = 2
                # minor ticks
                self.minor_tick_label_size = 14
                self.minor_ticks_size = 12
                self.minor_ticks_width = 1
                # scientific notation
                self.scientific_notation_size = self.font_size
                # color map
                self.color_map = "jet"
                # legend
                self.ticks_max_number_of_intervals = 255
                self.num_cbar_ticks = 7
                # resolution of results
                self.num_points_per_edge = 128
                # grid interpolation
                self.interpolation_method = "nearest"
                # save options
                self.dpi = 300
                self.file_format = "pdf"

        plot_config = PlotterConfigData()

        coordinates_np = calibration_data.inputs.detach().cpu().numpy()
        displacements_np = calibration_data.outputs.detach().cpu().numpy()

        def plot_one_displacements(
            coordinates: NPArray,
            displacements: NPArray,
            dimension: str,
            plot_config: PlotterConfigData,
        ) -> None:
            coordinates_x = coordinates[:, 0]
            coordinates_y = coordinates[:, 1]
            min_coordinates_x = np.nanmin(coordinates_x)
            max_coordinates_x = np.nanmax(coordinates_x)
            min_coordinates_y = np.nanmin(coordinates_y)
            max_coordinates_y = np.nanmax(coordinates_y)
            min_displacement = np.nanmin(displacements)
            max_displacement = np.nanmax(displacements)

            # grid data
            num_points_per_grid_dim = 128
            grid_coordinates_x = np.linspace(
                min_coordinates_x,
                max_coordinates_x,
                num=num_points_per_grid_dim,
            )
            grid_coordinates_y = np.linspace(
                min_coordinates_y,
                max_coordinates_y,
                num=num_points_per_grid_dim,
            )
            coordinates_grid_x, coordinates_grid_y = np.meshgrid(
                grid_coordinates_x, grid_coordinates_y
            )

            # interpolation
            displacements_grid = griddata(
                coordinates,
                displacements,
                (coordinates_grid_x, coordinates_grid_y),
                method=plot_config.interpolation_method,
            )

            figure, axes = plt.subplots()

            # figure size
            fig_height = 4
            box_length = 80
            box_height = 20
            figure.set_figheight(fig_height)
            figure.set_figwidth((box_length / box_height) * fig_height + 1)

            # title and labels
            title = f"Displacements {dimension}"
            axes.set_title(title, pad=plot_config.title_pad, **plot_config.font)
            axes.set_xlabel(plot_config.x_label, **plot_config.font)
            axes.set_ylabel(plot_config.y_label, **plot_config.font)
            axes.tick_params(
                axis="both", which="minor", labelsize=plot_config.minor_tick_label_size
            )
            axes.tick_params(
                axis="both", which="major", labelsize=plot_config.major_tick_label_size
            )

            # ticks
            num_coordinate_ticks = 3
            x_ticks = np.linspace(
                min_coordinates_x,
                max_coordinates_x,
                num=num_coordinate_ticks,
                endpoint=True,
            )
            y_ticks = np.linspace(
                min_coordinates_y,
                max_coordinates_y,
                num=num_coordinate_ticks,
                endpoint=True,
            )
            axes.set_xlim(min_coordinates_x, max_coordinates_x)
            axes.set_ylim(min_coordinates_y, max_coordinates_y)
            axes.set_xticks(x_ticks)
            axes.set_xticklabels(map(str, x_ticks.round(decimals=2)))
            axes.set_yticks(y_ticks)
            axes.set_yticklabels(map(str, y_ticks.round(decimals=2)))
            axes.tick_params(axis="both", which="major", pad=15)

            # normalizer
            tick_values = MaxNLocator(
                nbins=plot_config.ticks_max_number_of_intervals
            ).tick_values(min_displacement, max_displacement)
            normalizer = BoundaryNorm(
                tick_values, ncolors=plt.get_cmap(plot_config.color_map).N, clip=True
            )

            # plot
            plot = axes.pcolormesh(
                coordinates_grid_x,
                coordinates_grid_y,
                displacements_grid,
                cmap=plt.get_cmap(plot_config.color_map),
                norm=normalizer,
            )

            # color bar
            color_bar_ticks = (
                np.linspace(
                    min_displacement,
                    max_displacement,
                    num=plot_config.num_cbar_ticks,
                    endpoint=True,
                )
                .round(decimals=4)
                .tolist()
            )
            cbar = figure.colorbar(mappable=plot, ax=axes, ticks=color_bar_ticks)
            cbar.ax.set_yticklabels(map(str, color_bar_ticks))
            cbar.ax.minorticks_off()

            # hole
            origin_x = geometry_config.origin_x
            origin_y = geometry_config.origin_y
            radius_hole = geometry_config.plate_hole_radius
            plate_hole = plt.Circle(
                (origin_x, origin_y),
                radius=radius_hole,
                color="white",
            )
            axes.add_patch(plate_hole)

            # save
            file_name = (
                f"measurement_data_dispalcements_{dimension}.{plot_config.file_format}"
            )
            save_path = project_directory.create_output_file_path(
                file_name, output_subdirectory
            )
            dpi = 300
            figure.savefig(
                save_path,
                format=plot_config.file_format,
                bbox_inches="tight",
                dpi=dpi,
            )

        displacements_np_x = displacements_np[:, 0]
        displacements_np_y = displacements_np[:, 1]

        plot_one_displacements(coordinates_np, displacements_np_x, "x", plot_config)
        plot_one_displacements(coordinates_np, displacements_np_y, "y", plot_config)

    name_model_parameters_file = "model_parameters"
    model = load_model(
        model=ansatz,
        name_model_parameters_file=name_model_parameters_file,
        input_subdir=output_subdirectory,
        project_directory=project_directory,
        device=device,
    )

    calibration_data = generate_calibration_data()
    visualize_data(calibration_data)

    if consider_model_error:
        likelihood = create_standard_ppinn_q_likelihood_for_noise(
            model=model,
            num_model_parameters=num_material_parameters,
            data=calibration_data,
            make_robust=False,
            device=device,
        )
        output_subdir_calibration = os.path.join(
            output_subdirectory, "calibration_with_model_error"
        )
    else:
        likelihood = create_standard_ppinn_likelihood_for_noise(
            model=model,
            num_model_parameters=num_material_parameters,
            data=calibration_data,
            device=device,
        )
        output_subdir_calibration = os.path.join(
            output_subdirectory, "calibration_without_model_error"
        )

    def set_up_least_squares_config(
        data: CalibrationData,
    ) -> LeastSquaresConfig:
        mean_displacements = torch.mean(torch.absolute(data.outputs), dim=0)
        residual_weights = (
            (1 / mean_displacements).to(device).repeat((num_data_points, 1)).ravel()
        )
        return LeastSquaresConfig(
            ansatz=model,
            calibration_data=data,
            initial_parameters=initial_parameters,
            num_iterations=1000,
            resdiual_weights=residual_weights,
        )

    def set_up_metropolis_hastings_config(
        likelihood: Likelihood,
    ) -> MetropolisHastingsConfig:
        std_proposal_density_bulk_modulus = 100.0
        std_proposal_density_shear_modulus = 50.0
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
        return MetropolisHastingsConfig(
            likelihood=likelihood,
            prior=prior,
            initial_parameters=initial_parameters,
            num_iterations=int(1e5),
            num_burn_in_iterations=int(1e4),
            cov_proposal_density=cov_proposal_density,
        )

    def set_up_hamiltonian_configs(
        likelihood: Likelihood,
    ) -> HamiltonianConfig:
        return HamiltonianConfig(
            likelihood=likelihood,
            prior=prior,
            initial_parameters=initial_parameters,
            num_iterations=int(1e4),
            num_burn_in_iterations=int(5e3),
            num_leabfrog_steps=256,
            leapfrog_step_sizes=torch.tensor([1, 1], device=device),
        )

    def set_up_efficient_nuts_configs_configs(
        likelihood: Likelihood,
    ) -> EfficientNUTSConfig:
        return EfficientNUTSConfig(
            likelihood=likelihood,
            prior=prior,
            initial_parameters=initial_parameters,
            num_iterations=int(1e4),
            num_burn_in_iterations=int(5e3),
            max_tree_depth=8,
            leapfrog_step_sizes=torch.tensor([1, 1], device=device),
        )

    if use_least_squares:
        configs_ls = set_up_least_squares_config(calibration_data)
        start = perf_counter()
        test_least_squares_calibration(
            calibration_configs=(configs_ls,),
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
        configs_mh = set_up_metropolis_hastings_config(likelihood)
        start = perf_counter()
        test_coverage(
            calibration_configs=(configs_mh,),
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
        configs_h = set_up_hamiltonian_configs(likelihood)
        start = perf_counter()
        test_coverage(
            calibration_configs=(configs_h,),
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
        configs_en = set_up_efficient_nuts_configs_configs(likelihood)
        start = perf_counter()
        test_coverage(
            calibration_configs=(configs_en,),
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