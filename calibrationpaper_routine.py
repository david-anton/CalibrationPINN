import os
import sys

import pandas as pd
import torch

from parametricpinn.ansatz import create_standard_normalized_hbc_ansatz_2D
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader, PandasDataWriter
from parametricpinn.network import FFNN
from parametricpinn.settings import Settings, get_device
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
# Input paths
input_dir_calibration_data = "Paper_Calibration"
input_subdir_high_noise = "with_noise_4e-04"
input_subdir_low_noise = "with_noise_2e-04"
input_file_high_noise = "displacements_withNoise4e-04.csv"
input_file_low_noise = "displacements_withNoise2e-04.csv"
# Output path
label_ux = "u_x"
label_uy = "u_y"
output_subdir_name = "Paper_Calibration"
output_file_name = "displacements"


settings = Settings()
settings.INPUT_SUBDIR = "input"
settings.OUTPUT_SUBDIR = "output"
project_directory = ProjectDirectory(settings)
device = get_device()
data_writer = PandasDataWriter(project_directory)


min_inputs = torch.tensor([-edge_length, 0.0, min_youngs_modulus, min_poissons_ratio])
max_inputs = torch.tensor([0.0, edge_length, max_youngs_modulus, max_poissons_ratio])
min_outputs = torch.tensor([0.0, 0.0])
max_outputs = torch.tensor([0.0, 0.0])


def create_ansatz() -> Module:
    network = FFNN(layer_sizes=layer_sizes)
    return create_standard_normalized_hbc_ansatz_2D(
        displacement_x_right=torch.tensor(0.0).to(device),
        displacement_y_bottom=torch.tensor(0.0).to(device),
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        min_outputs=min_outputs,
        max_outputs=max_outputs,
        network=network,
    ).to(device)


def load_calibration_data() -> tuple[Tensor, Tensor]:
    noise_level = str(sys.argv[3])
    if noise_level == "low":
        input_subdir = input_subdir_low_noise
        input_file_name = input_file_low_noise
    elif noise_level == "high":
        input_subdir = input_subdir_high_noise
        input_file_name = input_file_high_noise
    else:
        raise SystemExit("Passed noise level not defined.")

    input_subdir_path = os.path.join(input_dir_calibration_data, input_subdir)
    data_reader = CSVDataReader(project_directory)
    data = torch.from_numpy(data_reader.read(input_file_name, input_subdir_path))
    coordinates = data[:, :2]
    noisy_displacements = data[:, 2:]

    return coordinates, noisy_displacements


youngs_modulus = float(sys.argv[1])
poissons_ratio = float(sys.argv[2])
parameters = torch.tensor(
    [youngs_modulus, poissons_ratio], dtype=torch.float64, device=device
)

ansatz = create_ansatz()

coordinates, _ = load_calibration_data()
num_data_points = coordinates.size()[0]
model_inputs = torch.concat(
    (
        coordinates,
        parameters.repeat((num_data_points, 1)),
    ),
    dim=1,
).to(device)

displacements = ansatz(model_inputs)
displacements_pdf = pd.DataFrame(
    data=displacements.detach().cpu().numpy(), columns=[label_ux, label_uy]
)
data_writer.write(
    displacements_pdf,
    subdir_name=output_subdir_name,
    file_name=output_file_name,
    header=True,
)
