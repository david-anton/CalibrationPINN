import os
from datetime import date

import numpy as np
import pandas as pd
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_quarter_plate_with_hole,
)
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
from parametricpinn.fem import (
    LinearElasticityProblemConfig_K_G,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    generate_simulation_data,
    run_simulation,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.network import FFNN
from parametricpinn.postprocessing.plot import (
    DisplacementsPlotterConfig2D,
    plot_displacements_2d,
)
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.training.training_standard_linearelasticity_quarterplatewithhole import (
    TrainingConfiguration,
    train_parametric_pinn,
)
from parametricpinn.types import Tensor

### Configuration
# Set up
use_stress_bc = True
material_model = "plane stress"
num_material_parameters = 2
edge_length = 100.0
radius = 10.0
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
bulk_modulus = 175000.0
shear_modulus = 80769.0
# Network
layer_sizes = [4, 64, 64, 64, 64, 64, 64, 2]
activation = torch.nn.Tanh()
# Ansatz
distance_function = "normalized linear"
# Training
num_collocation_points = 8192
num_points_per_bc = 256
bcs_overlap_distance = 1e-2
bcs_overlap_angle_distance = 1e-2
number_training_epochs = 5000
weight_pde_loss = 1.0
weight_traction_bc_loss = 1.0
if use_stress_bc:
    weight_stress_bc_loss = 1.0
else:
    weight_stress_bc_loss = 0.0
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size = 0.1
# Validation
regenerate_valid_data = True
validation_interval = 1
num_points_valid = 2048
# Input/output
input_subdir_valid = f"20240503_validation_data_linearelasticity_quarterplatewithhole_K_{bulk_modulus}_G_{shear_modulus}_edge_{int(edge_length)}_radius_{int(radius)}_traction_{int(traction_left_x)}_elementsize_{fem_element_size}"
current_date = date.today().strftime("%Y%m%d")
output_date = current_date
output_subdirectory = f"{output_date}_parametric_pinns_calibration_paper_bcscomparison"
if use_stress_bc:
    output_subdir_training = os.path.join(
        output_subdirectory, "training", "with_stress_bc"
    )
else:
    output_subdir_training = os.path.join(
        output_subdirectory, "training", "without_stress_bc"
    )
output_subdir_normalization = os.path.join(output_subdir_training, "normalization")
save_metadata = True


### Set up simulation
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
    tuple[
        QuarterPlateWithHoleTrainingDataset2D,
        SimulationDataset2D | None,
        SimulationDataset2D,
    ]
):
    def _create_pinn_training_dataset() -> QuarterPlateWithHoleTrainingDataset2D:
        print("Generate training data ...")
        parameters_samples = torch.tensor(
            [[bulk_modulus, shear_modulus]], device=device
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

    def _create_validation_dataset() -> SimulationDataset2D:
        def _generate_validation_data() -> None:
            domain_config = create_fem_domain_config()
            problem_config = LinearElasticityProblemConfig_K_G(
                model=material_model,
                material_parameters=(bulk_modulus, shear_modulus),
            )
            generate_simulation_data(
                domain_config=domain_config,
                problem_configs=[problem_config],
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
            num_samples=1,
            project_directory=project_directory,
        )
        return create_simulation_dataset(config_validation_data)

    training_dataset_pinn = _create_pinn_training_dataset()
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

    def _print_normalization_values(normalization_values: dict[str, Tensor]) -> None:
        print("###########################")
        print(f"Min inputs {normalization_values[key_min_inputs]}")
        print(f"Max inputs {normalization_values[key_max_inputs]}")
        print(f"Min outputs {normalization_values[key_min_outputs]}")
        print(f"Max outputs {normalization_values[key_max_outputs]}")
        print("###########################")

    def _determine_normalization_values() -> dict[str, Tensor]:
        min_coordinate_x = -edge_length
        max_coordinate_x = 0.0
        min_coordinate_y = 0.0
        max_coordinate_y = edge_length
        min_coordinates = torch.tensor([min_coordinate_x, min_coordinate_y])
        max_coordinates = torch.tensor([max_coordinate_x, max_coordinate_y])

        min_parameters = torch.tensor([bulk_modulus, shear_modulus])
        max_parameters = torch.tensor([bulk_modulus, shear_modulus])

        min_inputs = torch.concat((min_coordinates, min_parameters))
        max_inputs = torch.concat((max_coordinates, max_parameters))

        print("Run FE simulations to determine normalization values ...")
        domain_config = create_fem_domain_config()
        problem_config = LinearElasticityProblemConfig_K_G(
            model=material_model,
            material_parameters=(bulk_modulus, shear_modulus),
        )
        simulation_config = SimulationConfig(
            domain_config=domain_config,
            problem_config=problem_config,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
        )
        results_output_subdir = os.path.join(
            output_subdir_normalization, "fem_simulation_results_displacements"
        )
        simulation_results = run_simulation(
            simulation_config=simulation_config,
            save_results=True,
            save_metadata=True,
            output_subdir=results_output_subdir,
            project_directory=project_directory,
        )

        min_displacement_x = float(np.amin(simulation_results.displacements_x))
        max_displacement_x = float(np.amax(simulation_results.displacements_x))
        min_displacement_y = float(np.amin(simulation_results.displacements_y))
        max_displacement_y = float(np.amax(simulation_results.displacements_y))
        min_outputs = torch.tensor([min_displacement_x, min_displacement_y])
        max_outputs = torch.tensor([max_displacement_x, max_displacement_y])
        normalization_values = {
            key_min_inputs: min_inputs.to(device),
            key_max_inputs: max_inputs.to(device),
            key_min_outputs: min_outputs.to(device),
            key_max_outputs: max_outputs.to(device),
        }

        _print_normalization_values(normalization_values)
        _save_normalization_values(normalization_values)
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
        weight_data_loss=0.0,
        training_dataset_pinn=training_dataset_pinn,
        number_training_epochs=number_training_epochs,
        training_batch_size=1,
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
            (bulk_modulus, shear_modulus),
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


training_dataset_pinn, training_dataset_data, validation_dataset = create_datasets()
training_step()
