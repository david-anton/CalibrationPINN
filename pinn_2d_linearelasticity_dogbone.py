import math
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
    DogBoneGeometryConfig,
    DogBoneTrainingDataset2D,
    DogBoneTrainingDataset2DConfig,
    create_training_dataset,
)
from parametricpinn.data.validationdata_elasticity_2d import (
    ValidationDataset2D,
    ValidationDataset2DConfig,
    create_validation_dataset,
)
from parametricpinn.fem import (
    DogBoneDomainConfig,
    LinearElasticityProblemConfig,
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

# from parametricpinn.training.pinn_training_standard_linearelasticity_dogbone import (
#     TrainingConfiguration,
#     train_parametric_pinn,
# )
from parametricpinn.training.pinn_training_standard_linearelasticity_dogbone import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
# Set up
material_model = "plane stress"
num_material_parameters = 2
traction_right_x = 106.2629
traction_right_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 210000.0
max_youngs_modulus = min_youngs_modulus
min_poissons_ratio = 0.3
max_poissons_ratio = min_poissons_ratio
# Network
layer_sizes = [4, 32, 32, 32, 32, 2]
# Ansatz
distance_function = "normalized linear"
# Training
num_samples_per_parameter = 1
num_collocation_points = 8192
number_points_per_bc = 128
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 500
weight_pde_loss = 1.0
weight_traction_bc_loss = 1.0
weight_free_traction_bc_loss = 1.0
weight_dirichlet_bc_loss =0.0
overlap_distance_angle_bcs=10.0
# Validation
regenerate_valid_data = False
input_subdir_valid = (
    "20231206_validation_data_linearelasticity_dogbone_E_210k_nu_03_elementsize_01"
)
num_samples_valid = 1
validation_interval = 1
num_points_valid = 4096
batch_size_valid = num_samples_valid
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size = 0.1
# Output
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_pinn_linearelasticity_dogbone_E_210k_nu_03_col_8192_bc_128_PDE_traction_free_overlap_10"
output_subdirectory_preprocessing = f"{output_date}_preprocessing"
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)

geometry_config = DogBoneGeometryConfig()


def create_fem_domain_config() -> DogBoneDomainConfig:
    return DogBoneDomainConfig(
        geometry_config=geometry_config,
        traction_right_x=traction_right_x,
        traction_right_y=traction_right_y,
        element_family=fem_element_family,
        element_degree=fem_element_degree,
        element_size=fem_element_size,
    )


def create_datasets() -> tuple[DogBoneTrainingDataset2D, ValidationDataset2D]:
    def _create_training_dataset() -> DogBoneTrainingDataset2D:
        print("Generate training data ...")
        traction_right = torch.tensor([traction_right_x, traction_right_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        config_training_data = DogBoneTrainingDataset2DConfig(
            traction_right=traction_right,
            volume_force=volume_force,
            min_youngs_modulus=min_youngs_modulus,
            max_youngs_modulus=max_youngs_modulus,
            min_poissons_ratio=min_poissons_ratio,
            max_poissons_ratio=max_poissons_ratio,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=number_points_per_bc,
            num_samples_per_parameter=num_samples_per_parameter,
            overlap_distance_angle_bcs=overlap_distance_angle_bcs
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
        min_coordinate_x = -geometry_config.half_box_length
        max_coordinate_x = geometry_config.half_box_length
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
        print("###########################")
        print(f"Min inputs {min_inputs}")
        print(f"Max inputs {max_inputs}")
        print(f"Min outputs {min_outputs}")
        print(f"Max outputs {max_outputs}")
        print("###########################")
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
            [-geometry_config.half_box_length], device=device
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
    area_dogbone = (geometry_config.box_length * geometry_config.box_height) - (
        (2 * geometry_config.parallel_length * geometry_config.parallel_height)
        + (
            (4 * geometry_config.angle_max_tapered / 360)
            * math.pi
            * geometry_config.tapered_radius**2
        )
        + (math.pi * geometry_config.plate_hole_radius**2)
    )

    train_config = TrainingConfiguration(
        ansatz=ansatz,
        material_model=material_model,
        area_dogbone=area_dogbone,
        num_points_per_bc=number_points_per_bc,
        weight_pde_loss=weight_pde_loss,
        weight_traction_bc_loss=weight_traction_bc_loss,
        weight_free_traction_bc_loss=weight_free_traction_bc_loss,
        weight_dirichlet_bc_loss=weight_dirichlet_bc_loss,
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


training_dataset, validation_dataset = create_datasets()
training_step()
