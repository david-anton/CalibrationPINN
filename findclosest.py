import torch
from calibrationpinn.data.parameterssampling import sample_quasirandom_sobol
from calibrationpinn.settings import get_device, set_default_dtype, set_seed
from calibrationpinn.types import Tensor

device = get_device()
set_default_dtype(torch.float64)
set_seed(0)


def find_closest_sample(parameters: Tensor, parameters_samples: Tensor) -> Tensor:
    euclidean_norm = torch.sqrt(
        torch.sum((parameters_samples - parameters) ** 2, dim=1)
    )
    index_argmin = torch.argmin(euclidean_norm)
    return parameters_samples[index_argmin]


min_bulk_modulus = 100000.0
max_bulk_modulus = 200000.0
min_shear_modulus = 60000.0
max_shear_modulus = 100000.0
num_parameter_samples_data = 128


parameters_samples = sample_quasirandom_sobol(
    min_parameters=[min_bulk_modulus, min_shear_modulus],
    max_parameters=[max_bulk_modulus, max_shear_modulus],
    num_samples=num_parameter_samples_data,
    device=device,
)


identified_parameters_raw = torch.tensor([109343.0, 71125.0])
closest_training_sample_raw = find_closest_sample(
    identified_parameters_raw, parameters_samples
)
identified_parameters_interpolated = torch.tensor([126679.0, 73444.0])
closest_training_sample_interpolated = find_closest_sample(
    identified_parameters_interpolated, parameters_samples
)

print("############################################################")
print("All parameter samples:")
print(parameters_samples)
print("############################################################")
print("Raw data:")
print(f"Identified parameters: {identified_parameters_raw}")
print(f"Closest parameters sample: {closest_training_sample_raw}")
print("############################################################")
print("Interpolated data:")
print(f"Identified parameters: {identified_parameters_interpolated}")
print(f"Closest parameters sample: {closest_training_sample_interpolated}")
