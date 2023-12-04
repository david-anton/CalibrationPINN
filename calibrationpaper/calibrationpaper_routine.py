############################################################
# TO DO
############################################################
# 1) Copy model parameters in input/Paper_Calibration.
# 2) Set project directory path in settings object.
############################################################
############################################################

import os
import sys
from pathlib import Path

import pandas as pd
import torch

from parametricpinn.ansatz import (
    create_standard_normalized_hbc_ansatz_quarter_plate_with_hole,
)
from parametricpinn.calibration.utility import load_model
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader, PandasDataWriter
from parametricpinn.network import FFNN
from parametricpinn.settings import Settings, get_device, set_default_dtype, set_seed
from parametricpinn.types import Module, Tensor

### Configuration
# Set up
edge_length = 10.0
min_youngs_modulus = 180000.0
max_youngs_modulus = 240000.0
min_poissons_ratio = 0.2
max_poissons_ratio = 0.4
# Network
layer_sizes = [4, 32, 32, 32, 32, 2]
name_model_parameters_file = "model_parameters"
# Input paths
input_dir_calibration_data = "Paper_Calibration"
input_subdir_high_noise = "with_noise_4e-04"
input_subdir_low_noise = "with_noise_2e-04"
input_subdir_clean = "without_noise"
input_file_high_noise = "displacements_withNoise4e-04.csv"
input_file_low_noise = "displacements_withNoise2e-04.csv"
input_file_clean = "displacements_withoutNoise.csv"
# Output path
label_ux = "u_x"
label_uy = "u_y"
output_subdir_name = "Paper_Calibration"
output_file_name = "displacements"


youngs_modulus = float(sys.argv[1])
poissons_ratio = float(sys.argv[2])
noise_level = str(sys.argv[3])


settings = Settings()
######################################################
######################################################
settings.PROJECT_DIR = Path("/workspaces/app")
######################################################
######################################################
settings.INPUT_SUBDIR = "input"
settings.OUTPUT_SUBDIR = "output"
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)
data_writer = PandasDataWriter(project_directory)


min_inputs = torch.tensor([-edge_length, 0.0, min_youngs_modulus, min_poissons_ratio])
max_inputs = torch.tensor([0.0, edge_length, max_youngs_modulus, max_poissons_ratio])
min_outputs = torch.tensor([-0.0102, -0.0045])
max_outputs = torch.tensor([0.0, 9.3976e-08])


def create_ansatz() -> Module:
    network = FFNN(layer_sizes=layer_sizes)
    return create_standard_normalized_hbc_ansatz_quarter_plate_with_hole(
        displacement_x_right=torch.tensor(0.0).to(device),
        displacement_y_bottom=torch.tensor(0.0).to(device),
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
        network=network,
        device=device,
    ).to(device)


def load_parameters(ansatz: Module) -> Module:
    return load_model(
        model=ansatz,
        name_model_parameters_file=name_model_parameters_file,
        input_subdir=input_dir_calibration_data,
        project_directory=project_directory,
        device=device,
        load_from_output_dir=False,
    )


def load_coordinates_data() -> Tensor:
    if noise_level == "low":
        input_subdir = input_subdir_low_noise
        input_file_name = input_file_low_noise
    elif noise_level == "high":
        input_subdir = input_subdir_high_noise
        input_file_name = input_file_high_noise
    elif noise_level == "clean":
        input_subdir = input_subdir_clean
        input_file_name = input_file_clean
    else:
        raise SystemExit("Passed noise level not defined.")

    input_subdir_path = os.path.join(input_dir_calibration_data, input_subdir)
    data_reader = CSVDataReader(project_directory)
    data = torch.from_numpy(data_reader.read(input_file_name, input_subdir_path))
    coordinates = data[:, :2]

    return coordinates


def modifiy_coordinates(coordinates: Tensor) -> Tensor:
    return coordinates - torch.tensor([edge_length, 0.0], dtype=torch.float64).repeat(
        (num_data_points, 1)
    )


def write_displacements(displacements: Tensor) -> None:
    displacements_pdf = pd.DataFrame(
        data=displacements.detach().cpu().numpy(), columns=[label_ux, label_uy]
    )
    data_writer.write(
        displacements_pdf,
        subdir_name=output_subdir_name,
        file_name=output_file_name,
        header=True,
    )


parameters = torch.tensor(
    [youngs_modulus, poissons_ratio], dtype=torch.float64, device=device
)

ansatz = create_ansatz()
ansatz = load_parameters(ansatz)

coordinates = load_coordinates_data()
num_data_points = coordinates.size()[0]
coordinates = modifiy_coordinates(coordinates)

model_inputs = torch.concat(
    (
        coordinates,
        parameters.repeat((num_data_points, 1)),
    ),
    dim=1,
).to(device)

displacements = ansatz(model_inputs)
write_displacements(displacements)
