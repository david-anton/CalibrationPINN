from datetime import date

import numpy as np
import torch

from parametricpinn.fem import (
    LinearElasticityProblemConfig,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    run_simulation,
)
from parametricpinn.fem.convergence import calculate_l2_error
from parametricpinn.fem.problems import SimulationResults
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
fem_mesh_resolution_reference = 0.1
fem_mesh_resolution_factors = np.array([8, 4, 2, 1])
fem_mesh_resolution_tests = list(
    fem_mesh_resolution_factors * fem_mesh_resolution_reference
)
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


def calculate_approximate_solution(mesh_resolution) -> SimulationResults:
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
    return run_simulation(
        simulation_config=simulation_config,
        save_results=False,
        save_metadata=False,
        output_subdir=output_subdirectory,
        project_directory=project_directory,
    )


reference_solution = calculate_approximate_solution(fem_mesh_resolution_reference)
u_exact = reference_solution.function


mesh_resolution_record = []
l2_error_record = []
num_dofs_record = []
for mesh_resolution in fem_mesh_resolution_tests:
    approximate_solution = calculate_approximate_solution(mesh_resolution)
    u_approx = approximate_solution.function
    l2_error = calculate_l2_error(u_approx, u_exact)
    num_dofs = approximate_solution.coordinates_x.size
    mesh_resolution_record.append(mesh_resolution)
    l2_error_record.append(l2_error)
    num_dofs_record.append(num_dofs)

for mesh_resolution, l2_error, num_dofs in zip(
    mesh_resolution_record, l2_error_record, num_dofs_record
):
    print(
        f"Mesh: resolution: {mesh_resolution} \t L2 error: {l2_error} \t Number DOFs: {num_dofs}"
    )
