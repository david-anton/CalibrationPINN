from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from parametricpinn.fem import (
    LinearElasticityProblemConfig,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    run_simulation,
)
from parametricpinn.fem.base import DFunction
from parametricpinn.fem.convergence import (
    calculate_infinity_error,
    calculate_l2_error,
    calculate_relative_l2_error,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.settings import Settings, set_default_dtype

### Configuration
# Set up
material_model = "plane stress"
edge_length = 100.0
radius = 10.0
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
youngs_modulus = 210000.0
poissons_ratio = 0.3
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_size_tests = [3.2, 1.6, 0.8, 0.4, 0.2, 0.1]
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdirectory = (
    f"{current_date}_convergence_study_linearelasticity_quarterplatewithhole"
)

# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
set_default_dtype(torch.float64)


def create_fem_domain_config(
    element_size: float,
) -> QuarterPlateWithHoleDomainConfig:
    return QuarterPlateWithHoleDomainConfig(
        edge_length=edge_length,
        radius=radius,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        element_family=fem_element_family,
        element_degree=fem_element_degree,
        element_size=element_size,
    )


def calculate_approximate_solution(element_size) -> DFunction:
    domain_config = create_fem_domain_config(element_size)
    problem_config = LinearElasticityProblemConfig(
        model=material_model,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
    )
    simulation_config = SimulationConfig(
        domain_config=domain_config,
        problem_config=problem_config,
        volume_force_x=volume_force_x,
        volume_force_y=volume_force_y,
    )
    results = run_simulation(
        simulation_config=simulation_config,
        save_results=False,
        save_metadata=False,
        output_subdir=output_subdirectory,
        project_directory=project_directory,
    )
    return results.function


u_exact = calculate_approximate_solution(fem_element_size_tests[-1])

element_size_record: list[float] = []
l2_error_record: list[float] = []
relative_l2_error_record: list[float] = []
inifinity_error_record: list[float] = []
num_dofs_record: list[int] = []
num_elements_record: list[int] = []


print("Start convergence analysis")
for element_size in fem_element_size_tests:
    u_approx = calculate_approximate_solution(element_size)
    l2_error = calculate_l2_error(u_approx, u_exact).item()
    relative_l2_error = calculate_relative_l2_error(u_approx, u_exact).item()
    infinity_error = calculate_infinity_error(u_approx, u_exact).item()
    num_elements = u_approx.function_space.mesh.topology.index_map(2).size_global
    num_dofs = u_approx.function_space.tabulate_dof_coordinates().size
    element_size_record.append(element_size)
    l2_error_record.append(l2_error)
    relative_l2_error_record.append(relative_l2_error)
    inifinity_error_record.append(infinity_error)
    num_elements_record.append(num_elements)
    num_dofs_record.append(num_dofs)

# Save results
results_frame = pd.DataFrame(
    {
        "element_size": element_size_record,
        "l2 error": l2_error_record,
        "relative l2 error": relative_l2_error_record,
        "infinity error": inifinity_error_record,
        "number elements": num_elements_record,
        "number dofs": num_dofs_record,
    }
)
pandas_data_writer = PandasDataWriter(project_directory)
pandas_data_writer.write(
    data=results_frame,
    file_name="results",
    subdir_name=output_subdirectory,
    header=True,
)

############################################################
print("Preprocessing")


# Plot log l2 error
figure, axes = plt.subplots()
log_element_size = np.log(np.array(element_size_record))
log_l2_error = np.log(np.array(l2_error_record))

slope, intercept, _, _, _ = stats.linregress(log_element_size, log_l2_error)
regression_log_element_size = log_element_size
regression_log_l2_error = slope * regression_log_element_size + intercept

axes.plot(log_element_size, log_l2_error, "ob", label="simulation")
axes.plot(
    regression_log_element_size, regression_log_l2_error, "--r", label="regression"
)
axes.set_xlabel("log element size")
axes.set_ylabel("log l2 error")
axes.set_title("Convergence")
axes.legend(loc="best")
axes.text(
    log_element_size[2],
    log_l2_error[-1],
    f"convergence rate: {slope:.6}",
    style="italic",
    bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
)
file_name = f"convergence_log_l2_error.png"
output_path = project_directory.create_output_file_path(file_name, output_subdirectory)
figure.savefig(output_path, bbox_inches="tight", dpi=256)
plt.clf()


# Plot log infinity error
figure, axes = plt.subplots()
log_element_size = np.log(np.array(element_size_record))
log_infinity_error = np.log(np.array(inifinity_error_record))

slope, intercept, _, _, _ = stats.linregress(log_element_size, log_infinity_error)
regression_log_element_size = log_element_size
regression_log_infinity_error = slope * regression_log_element_size + intercept

axes.plot(log_element_size, log_infinity_error, "ob", label="simulation")
axes.plot(
    regression_log_element_size, regression_log_l2_error, "--r", label="regression"
)
axes.set_xlabel("log element size")
axes.set_ylabel("log infinity error")
axes.legend(loc="best")
axes.text(
    log_element_size[2],
    log_infinity_error[-1],
    f"convergence rate: {slope:.6}",
    style="italic",
    bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
)
axes.set_title("Convergence")
file_name = f"convergence_log_infinity_error.png"
output_path = project_directory.create_output_file_path(file_name, output_subdirectory)
figure.savefig(output_path, bbox_inches="tight", dpi=256)
plt.clf()
