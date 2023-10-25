from datetime import date

import torch

from parametricpinn.fem import (
    NeoHookeanProblemConfig,
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
from parametricpinn.settings import Settings, set_default_dtype

### Configuration
# Set up
edge_length = 100.0
radius = 10.0
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
youngs_modulus = 2000.0
poissons_ratio = 0.3
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 2
fem_element_size_tests = [3.2, 1.6, 0.8, 0.4, 0.2, 0.1]
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdirectory = (
    f"{current_date}_convergence_study_neohookean_quarterplatewithhole"
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
    problem_config = NeoHookeanProblemConfig(
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

for i in range(0, len(fem_element_size_tests)):
    u_approx = calculate_approximate_solution(fem_element_size_tests[i])
    l2_error = calculate_l2_error(u_approx, u_exact)
    relative_l2_error = calculate_relative_l2_error(u_approx, u_exact)
    infinity_error = calculate_infinity_error(u_approx, u_exact)
    mesh_resolution = fem_element_size_tests[i]
    print(
        f"Mesh resolution: {mesh_resolution}: \t L2 error: {l2_error:.4} \t rel. L2 error: {relative_l2_error:.4} \t infinity error: {infinity_error:.4}"
    )
