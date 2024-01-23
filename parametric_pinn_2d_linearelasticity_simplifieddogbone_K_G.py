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
from parametricpinn.training.loss_2d.momentum_linearelasticity_K_G import (
    calculate_K_from_E_and_nu_factory,
    calculate_G_from_E_and_nu,
    calculate_E_from_K_and_G_factory,
    calculate_nu_from_K_and_G_factory,
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
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig2D,
    plot_displacements_2d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_standard_linearelasticity_simplifieddogbone_K_G import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
retrain_parametric_pinn = False
# Set up
material_model = "plane stress"
num_material_parameters = 2
traction_right_x = 106.2629  # [N/mm^2]
traction_right_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 160000.0
max_youngs_modulus = 260000.0
min_poissons_ratio = 0.15
max_poissons_ratio = 0.45
# Network
layer_sizes = [4, 128, 128, 128, 128, 128, 128, 2]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_per_parameter = 32
num_collocation_points = 64
number_points_per_bc = 64
bcs_overlap_angle_distance_left = 1e-2
bcs_overlap_distance_parallel_right = 1e-2
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 30000
weight_pde_loss = 1.0
weight_traction_bc_loss = 1.0
weight_symmetry_bc_loss = 1e5
# Validation
regenerate_valid_data = False
input_subdir_valid = "20240119_validation_data_linearelasticity_simplifieddogbone_E_160k_260k_nu_015_045_elementsize_01_K_G"
num_samples_valid = 32
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
# Calibration
input_subdir_calibration = os.path.join(
    "Paper_PINNs", "20240123_experimental_dic_data_dogbone"
)
input_file_name_calibration = "displacements_dic.csv"
use_least_squares = True
use_random_walk_metropolis_hasting = True
use_hamiltonian = False
use_efficient_nuts = False
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size = 0.1
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = 20240119
output_subdirectory = f"{output_date}_parametric_pinn_linearelasticity_simplifieddogbone_E_160k_260k_nu_015_045_col_64_bc_64_neurons_6_128_K_G"
output_subdirectory_preprocessing = f"{output_date}_preprocessing"
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

geometry_config = SimplifiedDogBoneGeometryConfig()

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
        parameter_samples_K_G = torch.concat(
            (parameter_samples_K, parameter_samples_G), dim=1
        )

        traction_right = torch.tensor([traction_right_x, traction_right_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        config_training_data = SimplifiedDogBoneTrainingDataset2DConfig(
            parameters_samples=parameter_samples_K_G,
            traction_right=traction_right,
            volume_force=volume_force,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=number_points_per_bc,
            bcs_overlap_angle_distance_left=bcs_overlap_angle_distance_left,
            bcs_overlap_distance_parallel_right=bcs_overlap_distance_parallel_right,
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
            material_parameters=(significant_bulk_modulus, significant_shear_modulus),
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
        weight_symmetry_bc_loss=weight_symmetry_bc_loss,
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
        material_parameters_list_list = [
            (min_youngs_modulus, min_poissons_ratio),
            (min_youngs_modulus, max_poissons_ratio),
            (max_youngs_modulus, min_poissons_ratio),
            (max_youngs_modulus, max_poissons_ratio),
            (210000.0, 0.3),
        ]
        youngs_moduli, poissons_ratios = zip(*material_parameters_list_list)

        domain_config = create_fem_domain_config()
        problem_configs = []
        for i in range(len(material_parameters_list_list)):
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
    exact_youngs_modulus = 192800.0
    exact_poissons_ratio = 0.2491
    exact_bulk_modulus = calculate_K_from_E_and_nu(
        E=exact_youngs_modulus, nu=exact_poissons_ratio
    )
    exact_shear_modulus = calculate_G_from_E_and_nu(
        E=exact_youngs_modulus, nu=exact_poissons_ratio
    )
    num_data_points = 5475
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
        coordinate_shift_x = geometry_config.left_half_measurement_length
        coordinate_shift_y = geometry_config.half_measurement_height
        full_coordinates = full_shifted_coordinates - torch.tensor(
            [coordinate_shift_x, coordinate_shift_y], dtype=torch.float64
        )
        full_displacements = torch.from_numpy(data[:, slice_displacements]).type(
            torch.float64
        )
        random_indices = torch.randint(low=0, high=size_data, size=(num_data_points,))
        coordinates = full_coordinates[random_indices, :].to(device)
        displacements = full_displacements[random_indices, :].to(device)
        return coordinates, displacements

    def visualize_data(coordinates: Tensor, displacements: Tensor) -> None:
        # imports
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm
        from matplotlib.ticker import MaxNLocator
        from scipy.interpolate import griddata

        from parametricpinn.types import NPArray

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

        coordinates_np = coordinates.detach().cpu().numpy()
        displacements_np = displacements.detach().cpu().numpy()

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

    coordinates, displacements = generate_calibration_data()
    visualize_data(coordinates, displacements)
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

    mean_displacements = torch.mean(torch.absolute(displacements), dim=0)
    max_displacements = torch.amax(torch.absolute(displacements), dim=0)
    print(f"1 / mean displacements: {1 / mean_displacements}")
    print(f"1 / max displacements: {1 / max_displacements}")
    residual_weights = 1 / mean_displacements
    print(f"Used residual weights: {residual_weights}")

    least_squares_config = LeastSquaresConfig(
        ansatz=model,
        calibration_data=data,
        initial_parameters=initial_parameters,
        num_iterations=1000,
        resdiual_weights=residual_weights.to(device)
        .repeat((num_data_points, 1))
        .ravel(),
    )

    std_proposal_density_bulk_modulus = 1000
    std_proposal_density_shear_modulus = 500
    mcmc_config_mh = MetropolisHastingsConfig(
        likelihood=likelihood,
        prior=prior,
        initial_parameters=initial_parameters,
        num_iterations=int(1e4),
        num_burn_in_iterations=int(1e4),
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
        identified_K = identified_parameters[0]
        identified_G = identified_parameters[1]
        identified_E = calculate_E_from_K_and_G(K=identified_K, G=identified_G)
        identified_nu = calculate_nu_from_K_and_G(K=identified_K, G=identified_G)
        print(f"Identified parameters: K = {identified_K} and G = {identified_G}")
        print(f"Identified parameters: E = {identified_E} and nu = {identified_nu}")
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
