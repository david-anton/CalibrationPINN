############################################################
# TO DO
############################################################
# 1) Copy model the following files in the corresponding subdirectories:
#       -input/Paper_PINNs/training
#           - "model_parameters"
#       -input/Paper_PINNs/normalization
#           - "minimum_inputs.csv"
#           - "maximum_inputs.csv"
#           - "minimum_outputs.csv"
#           - "maximum_outputs.csv"
# 2) Set project directory path in settings object.
############################################################
############################################################

import os
import sys
from pathlib import Path

import pandas as pd
import torch

from parametricpinn.ansatz import (
    StandardAnsatz,
    create_standard_normalized_hbc_ansatz_clamped_left,
)
from parametricpinn.calibration.utility import load_model
from parametricpinn.data.trainingdata_2d import SimplifiedDogBoneGeometryConfig
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader, PandasDataWriter
from parametricpinn.network import FFNN
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.types import Tensor

### Settings
settings = Settings()
settings.PROJECT_DIR = Path("/workspaces/app")  # May need to be adjusted
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


### Configuration
input_subdir = output_subdir = "Paper_PINNs"
# Input
input_subdir_training = os.path.join(input_subdir, "training")
input_file_model_parameters = "model_parameters"
input_subdir_normalization = os.path.join(input_subdir, "normalization")
input_subdir_calibration_data = os.path.join(
    input_subdir, "20240123_experimental_dic_data_dogbone"
)
input_file_calibration_data = "displacements_dic.csv"
# Output
label_ux = "u_x"
label_uy = "u_y"
output_file_name = "displacements"


geometry_config = SimplifiedDogBoneGeometryConfig()


def create_ansatz() -> StandardAnsatz:
    key_min_inputs = "min_inputs"
    key_max_inputs = "max_inputs"
    key_min_outputs = "min_outputs"
    key_max_outputs = "max_outputs"
    file_name_min_inputs = "minimum_inputs.csv"
    file_name_max_inputs = "maximum_inputs.csv"
    file_name_min_outputs = "minimum_outputs.csv"
    file_name_max_outputs = "maximum_outputs.csv"

    def _read_normalization_values() -> dict[str, Tensor]:
        data_reader = CSVDataReader(project_directory)
        normalization_values = {}

        def _add_one_value_tensor(key: str, file_name: str) -> None:
            values = data_reader.read(
                file_name=file_name,
                subdir_name=input_subdir_normalization,
                read_from_output_dir=False,
            )

            normalization_values[key] = (
                torch.from_numpy(values[0]).type(torch.float64).to(device)
            )

        _add_one_value_tensor(key_min_inputs, file_name_min_inputs)
        _add_one_value_tensor(key_max_inputs, file_name_max_inputs)
        _add_one_value_tensor(key_min_outputs, file_name_min_outputs)
        _add_one_value_tensor(key_max_outputs, file_name_max_outputs)

        return normalization_values

    normalization_values = _read_normalization_values()
    layer_sizes = [4, 128, 128, 128, 128, 128, 128, 2]
    network = FFNN(layer_sizes=layer_sizes)
    distance_function = "normalized linear"
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


def load_parameters(ansatz: StandardAnsatz) -> StandardAnsatz:
    return load_model(
        model=ansatz,
        name_model_parameters_file=input_file_model_parameters,
        input_subdir=input_subdir_training,
        project_directory=project_directory,
        device=device,
        load_from_output_dir=False,
    )


def load_coordinates_data() -> Tensor:
    def _read_raw_data() -> tuple[Tensor, Tensor]:
        csv_reader = CSVDataReader(project_directory)
        data = csv_reader.read(
            file_name=input_file_calibration_data,
            subdir_name=input_subdir_calibration_data,
            read_from_output_dir=False,
        )
        slice_coordinates = slice(0, 2)
        slice_displacements = slice(2, 4)
        full_raw_coordinates = torch.from_numpy(data[:, slice_coordinates]).type(
            torch.float64
        )
        full_raw_displacements = torch.from_numpy(data[:, slice_displacements]).type(
            torch.float64
        )
        return full_raw_coordinates, full_raw_displacements

    def _transform_coordinates(full_raw_coordinates: Tensor) -> Tensor:
        coordinate_shift_x = geometry_config.left_half_measurement_length
        coordinate_shift_y = geometry_config.half_measurement_height
        return full_raw_coordinates - torch.tensor(
            [coordinate_shift_x, coordinate_shift_y], dtype=torch.float64
        )

    def _filter_data_points_within_measurement_area(
        full_raw_coordinates: Tensor, full_raw_displacements: Tensor
    ) -> tuple[Tensor, Tensor]:
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
        return full_coordinates, full_displacements

    full_raw_coordinates, full_raw_displacements = _read_raw_data()
    full_raw_coordinates = _transform_coordinates(full_raw_coordinates)
    (
        full_coordinates,
        full_displacements,
    ) = _filter_data_points_within_measurement_area(
        full_raw_coordinates, full_raw_displacements
    )
    return full_coordinates


def combine_inputs(coordinates: Tensor, material_parameters: Tensor) -> Tensor:
    num_data_points = len(coordinates)
    return torch.concat(
        (
            coordinates,
            material_parameters.repeat((num_data_points, 1)),
        ),
        dim=1,
    ).to(device)


def write_displacements(displacements: Tensor) -> None:
    data_writer = PandasDataWriter(project_directory)
    displacements_pdf = pd.DataFrame(
        data=displacements.detach().cpu().numpy(), columns=[label_ux, label_uy]
    )
    data_writer.write(
        displacements_pdf,
        subdir_name=output_subdir,
        file_name=output_file_name,
        header=True,
    )


bulk_modulus = float(sys.argv[1])
shear_modulus = float(sys.argv[2])
material_parameters = torch.tensor(
    [bulk_modulus, shear_modulus], dtype=torch.float64, device=device
)

ansatz = create_ansatz()
ansatz = load_parameters(ansatz)
coordinates = load_coordinates_data()
model_inputs = combine_inputs(coordinates, material_parameters)

displacements = ansatz(model_inputs)
write_displacements(displacements)
