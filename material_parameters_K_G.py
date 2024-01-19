from parametricpinn.data.parameterssampling import sample_uniform_grid
from parametricpinn.settings import get_device, Settings
from parametricpinn.training.loss_2d.momentumbase_linearelasticity import (
    calculate_G_from_E_and_nu,
    calculate_K_from_E_and_nu_factory,
)
from parametricpinn.io import ProjectDirectory
import matplotlib.pyplot as plt
from datetime import date
import numpy as np

device = get_device()
settings = Settings()
project_directory = ProjectDirectory(settings)
output_date = date.today().strftime("%Y%m%d")
output_subdirectory = f"{output_date}_distribution_of_K_and_G"

model = "plane stress"
min_E = 160000.0
max_E = 260000.0
min_nu = 0.15
max_nu = 0.45

params_E_nu = sample_uniform_grid(
    min_parameters=[min_E, min_nu],
    max_parameters=[max_E, max_nu],
    num_steps=[1000, 1000],
    device=device,
)
params_E = params_E_nu[:, 0].reshape(-1, 1)
params_nu = params_E_nu[:, 1].reshape(-1, 1)

calculate_K_from_E_and_nu = calculate_K_from_E_and_nu_factory(model)

params_K = calculate_K_from_E_and_nu(params_E, params_nu).detach().numpy()
params_G = calculate_G_from_E_and_nu(params_E, params_nu).detach().numpy()
print(f"Min K: {np.amin(params_K)}")
print(f"Max K: {np.amax(params_K)}")
print(f"Min G: {np.amin(params_G)}")
print(f"Max G: {np.amax(params_G)}")

print(calculate_K_from_E_and_nu(min_E, min_nu))
print(calculate_K_from_E_and_nu(max_E, max_nu))

print(calculate_G_from_E_and_nu(min_E, max_nu))
print(calculate_G_from_E_and_nu(max_E, min_nu))

### Plotting
num_bins = 1024

# Bulk modulus K
figure_K, axes_K = plt.subplots()
axes_K.set_title("Distribution of bulk modulus K")
axes_K.hist(
    params_K,
    bins=num_bins,
)
file_name = f"distribution_bulk_modulus_{model}.png"
output_path_K = project_directory.create_output_file_path(
    file_name=file_name, subdir_name=output_subdirectory
)
figure_K.savefig(output_path_K, bbox_inches="tight", dpi=300)
plt.clf()

# Bulk modulus K
figure_G, axes_G = plt.subplots()
axes_G.set_title("Distribution of shear modulus G")
axes_G.hist(
    params_G,
    bins=num_bins,
)
file_name = f"distribution_shear_modulus_{model}.png"
output_path_G = project_directory.create_output_file_path(
    file_name=file_name, subdir_name=output_subdirectory
)
figure_G.savefig(output_path_G, bbox_inches="tight", dpi=300)
plt.clf()
