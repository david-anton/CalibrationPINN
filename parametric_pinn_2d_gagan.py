import os
from datetime import date

import numpy as np
import torch

from parametricpinn.ansatz import create_normalized_hbc_ansatz_2D
from parametricpinn.data import (
    TrainingDataset2D,
    ValidationDataset2D,
    create_training_dataset_2D,
    create_validation_dataset_2D,
)
from parametricpinn.fem.platewithhole import (
    generate_validation_data as generate_validation_data,
)
from parametricpinn.fem.platewithhole import run_simulation
from parametricpinn.io import ProjectDirectory
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfigPWH,
    plot_displacements_pwh,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_2D import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Module, Tensor

### Configuration
retrain_parametric_pinn = True
# Set up
material_model = "plane stress"
edge_length = 100.0
radius = 10.0
traction_left_x = 0.0
traction_left_y = 100.0
volume_force_x = 0.0
volume_force_y = 0.0
min_youngs_modulus = 210000.0
max_youngs_modulus = 210000.0
min_poissons_ratio = 0.3
max_poissons_ratio = 0.3
# Network
layer_sizes = [4, 16, 16, 16, 16, 2]
# Training
num_samples_per_parameter = 1
num_collocation_points = 8192
number_points_per_bc = 1024
training_batch_size = num_samples_per_parameter**2
number_training_epochs = 40000
weight_pde_loss = 1.0
weight_symmetry_bc_loss = 1.0
weight_traction_bc_loss = 1.0
# Validation
regenerate_valid_data = True
input_subdir_valid = "20230727_validation_data_E_210k_nu_03_radius_10"
num_samples_valid = 1
validation_interval = 1
num_points_valid = 1024
batch_size_valid = num_samples_valid
fem_mesh_resolution = 0.1
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdirectory = f"{current_date}_test_case_Gagan_pinn_E_210k_nu_03_col_8192_bc_1024_neurons_4_16"
output_subdirectory_preprocessing = f"{current_date}_preprocessing"
save_metadata = True


# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def create_datasets() -> tuple[TrainingDataset2D, ValidationDataset2D]:
    def _create_training_dataset() -> TrainingDataset2D:
        print("Generate training data ...")
        traction_left = torch.tensor([traction_left_x, traction_left_y])
        volume_force = torch.tensor([volume_force_x, volume_force_y])
        return create_training_dataset_2D(
            edge_length=edge_length,
            radius=radius,
            traction_left=traction_left,
            volume_force=volume_force,
            min_youngs_modulus=min_youngs_modulus,
            max_youngs_modulus=max_youngs_modulus,
            min_poissons_ratio=min_poissons_ratio,
            max_poissons_ratio=max_poissons_ratio,
            num_collocation_points=num_collocation_points,
            num_points_per_bc=number_points_per_bc,
            num_samples_per_parameter=num_samples_per_parameter,
        )

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
            generate_validation_data(
                model=material_model,
                youngs_moduli=youngs_moduli,
                poissons_ratios=poissons_ratios,
                edge_length=edge_length,
                radius=radius,
                volume_force_x=volume_force_x,
                volume_force_y=volume_force_y,
                traction_left_x=traction_left_x,
                traction_left_y=traction_left_y,
                save_metadata=save_metadata,
                output_subdir=input_subdir_valid,
                project_directory=project_directory,
                mesh_resolution=fem_mesh_resolution,
            )

        print("Load validation data ...")
        if regenerate_valid_data:
            print("Run FE simulations to generate validation data ...")
            _generate_validation_data()
        return create_validation_dataset_2D(
            input_subdir=input_subdir_valid,
            num_points=num_points_valid,
            num_samples=num_samples_valid,
            project_directory=project_directory,
        )

    training_dataset = _create_training_dataset()
    validation_dataset = _create_validation_dataset()
    return training_dataset, validation_dataset

def create_ansatz() -> Module:
    def _determine_normalization_values() -> dict[str, Tensor]:
        min_coordinate_x = -edge_length
        max_coordinate_x = 0.0
        min_coordinate_y = 0.0
        max_coordinate_y = edge_length
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
        simulation_results = run_simulation(
            model=material_model,
            youngs_modulus=min_youngs_modulus,
            poissons_ratio=max_poissons_ratio,
            edge_length=edge_length,
            radius=radius,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            traction_left_x=traction_left_x,
            traction_left_y=traction_left_y,
            save_results=False,
            save_metadata=False,
            output_subdir=_output_subdir,
            project_directory=project_directory,
            mesh_resolution=fem_mesh_resolution,
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
    return create_normalized_hbc_ansatz_2D(
        displacement_x_right=torch.tensor(0.0).to(device),
        displacement_y_bottom=torch.tensor(0.0).to(device),
        network=network,
        min_inputs=normalization_values["min_inputs"],
        max_inputs=normalization_values["max_inputs"],
        min_outputs=normalization_values["min_outputs"],
        max_outputs=normalization_values["max_outputs"],
    ).to(device)


training_dataset, validation_dataset = create_datasets()
ansatz = create_ansatz()


def training_step() -> None:
    train_config = TrainingConfiguration(
        ansatz=ansatz,
        material_model=material_model,
        number_points_per_bc=number_points_per_bc,
        weight_pde_loss=weight_pde_loss,
        weight_symmetry_bc_loss=weight_symmetry_bc_loss,
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
        displacements_plotter_config = DisplacementsPlotterConfigPWH()

        plot_displacements_pwh(
            ansatz=ansatz,
            youngs_modulus_and_poissons_ratio_list=[
                (210000, 0.3),
                (183000, 0.27),
                (238000, 0.38),
            ],
            model=material_model,
            edge_length=edge_length,
            radius=radius,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            traction_left_x=traction_left_x,
            traction_left_y=traction_left_y,
            output_subdir=output_subdirectory,
            project_directory=project_directory,
            plot_config=displacements_plotter_config,
            device=device,
            mesh_resolution=0.5,
        )

    train_parametric_pinn(train_config=train_config)
    _plot_exemplary_displacement_fields()


def calibration_step() -> None:
    print("Start calibration ...")


if retrain_parametric_pinn:
    training_step()
calibration_step()
