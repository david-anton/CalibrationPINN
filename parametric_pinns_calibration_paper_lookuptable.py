import os

import torch

from calibrationpinn.io import ProjectDirectory
from calibrationpinn.io.readerswriters import CSVDataReader
from calibrationpinn.settings import Settings, get_device, set_default_dtype, set_seed
from calibrationpinn.types import Tensor

settings = Settings()
project_directory = ProjectDirectory(settings)
device = get_device()
set_default_dtype(torch.float64)
set_seed(0)
data_reader = CSVDataReader(project_directory)

num_training_samples = 128
num_valid_samples = 100

model = "linear_elasticity"

if model == "linear_elasticity":
    min_bulk_modulus = 100000.0
    max_bulk_modulus = 200000.0
    min_shear_modulus = 60000.0
    max_shear_modulus = 100000.0
    input_subdir_training = f"20240523_training_data_linearelasticity_quarterplatewithhole_K_{min_bulk_modulus}_{max_bulk_modulus}_G_{min_shear_modulus}_{max_shear_modulus}_edge_100_radius_10_traction_-100_elementsize_0.1"
    input_subdir_validation = f"20240523_validation_data_linearelasticity_quarterplatewithhole_K_{min_bulk_modulus}_{max_bulk_modulus}_G_{min_shear_modulus}_{max_shear_modulus}_edge_100_radius_10_traction_-100_elementsize_0.1"
elif model == "hyperelasticity":
    min_bulk_modulus = 4000.0
    max_bulk_modulus = 8000.0
    min_shear_modulus = 500.0
    max_shear_modulus = 1500.0
    input_subdir_training = f"20240523_training_data_neohooke_quarterplatewithhole_K_{int(min_bulk_modulus)}_{int(max_bulk_modulus)}_G_{int(min_shear_modulus)}_{int(max_shear_modulus)}_edge_100_radius_10_traction_-100_elementsize_0.2"
    input_subdir_validation = f"20240523_validation_data_neohooke_quarterplatewithhole_K_{int(min_bulk_modulus)}_{int(max_bulk_modulus)}_G_{int(min_shear_modulus)}_{int(max_shear_modulus)}_edge_100_radius_10_traction_-100_elementsize_0.2"


def read_training_samples() -> Tensor:
    return torch.vstack(
        [
            _read_one_parameter(i, input_subdir_training)
            for i in range(0, num_training_samples)
        ]
    )


def read_valid_samples() -> Tensor:
    return torch.vstack(
        [
            _read_one_parameter(i, input_subdir_validation)
            for i in range(0, num_valid_samples)
        ]
    )


def _read_one_parameter(idx_sample: int, input_directory: str) -> Tensor:
    file_name = "parameters"
    sample_subdir = f"sample_{idx_sample}"
    input_subdir = os.path.join(input_directory, sample_subdir)
    data = data_reader.read(file_name, input_subdir, read_from_output_dir=False)
    return torch.tensor(data, dtype=torch.get_default_dtype())


def find_closest_training_sample(
    valid_sample: Tensor, traninig_samples: Tensor
) -> Tensor:
    euclidean_norm = torch.sqrt(
        torch.sum((traninig_samples - valid_sample) ** 2, dim=1)
    )
    index_argmin = torch.argmin(euclidean_norm)
    return traninig_samples[index_argmin]


def calculate_absolute_relative_error(
    closest_training_sample: Tensor, valid_sample: Tensor
) -> Tensor:
    return torch.abs((closest_training_sample - valid_sample) / valid_sample) * 100


training_samples = read_training_samples()
valid_samples = read_valid_samples()

closest_training_samples = torch.vstack(
    [
        find_closest_training_sample(valid_sample, training_samples)
        for valid_sample in valid_samples
    ]
)

absolute_relative_errors = calculate_absolute_relative_error(
    closest_training_samples, valid_samples
)

mean_absolute_relative_error = torch.mean(absolute_relative_errors, dim=0)


print("############################################################")
print("Training parameter samples:")
print(training_samples)
print("############################################################")
print("Validation parameter samples:")
print(valid_samples)
print("############################################################")
print("Mean absolute relative error:")
print(mean_absolute_relative_error)
