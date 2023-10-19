from datetime import date

import torch

from parametricpinn.fem import (
    LinearElasticityProblemConfig,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    run_simulation,
)
from parametricpinn.fem.base import DFunction
from parametricpinn.fem.convergence import calculate_l2_error
from parametricpinn.io import ProjectDirectory
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
fem_mesh_resolution_tests = [3.2, 1.6, 0.8, 0.4, 0.1]
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
    mesh_resolution: float,
) -> QuarterPlateWithHoleDomainConfig:
    return QuarterPlateWithHoleDomainConfig(
        edge_length=edge_length,
        radius=radius,
        traction_left_x=traction_left_x,
        traction_left_y=traction_left_y,
        element_family=fem_element_family,
        element_degree=fem_element_degree,
        mesh_resolution=mesh_resolution,
    )


def calculate_approximate_solution(mesh_resolution) -> DFunction:
    domain_config = create_fem_domain_config(mesh_resolution)
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


u_approx = calculate_approximate_solution(fem_mesh_resolution_tests[0])

for i in range(1, len(fem_mesh_resolution_tests)):
    u_refined = calculate_approximate_solution(fem_mesh_resolution_tests[i])
    l2_error = calculate_l2_error(u_approx=u_approx, u_exact=u_refined)
    num_elements = u_approx.function_space.tabulate_dof_coordinates().size
    num_elements_refined = u_refined.function_space.tabulate_dof_coordinates().size
    print(f"{num_elements} \t -> {num_elements_refined}: \t L2 error ratio: {l2_error}")
    u_approx = u_refined
