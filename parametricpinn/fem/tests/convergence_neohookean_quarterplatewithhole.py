from datetime import date
from typing import Callable, Union

import numpy as np
import torch
import ufl
from dolfinx import fem
from mpi4py import MPI

from parametricpinn.fem import (
    NeoHookeanProblemConfig,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    run_simulation,
)
from parametricpinn.fem.base import DFunction
from parametricpinn.fem.problems import SimulationResults
from parametricpinn.io import ProjectDirectory
from parametricpinn.settings import Settings, set_default_dtype
from parametricpinn.types import NPArray

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
fem_mesh_resolution_reference = 0.5
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
    return run_simulation(
        simulation_config=simulation_config,
        save_results=False,
        save_metadata=False,
        output_subdir=output_subdirectory,
        project_directory=project_directory,
    )


def calculate_l2_error(
    u_approx: DFunction,
    u_exact: Union[DFunction, Callable[[NPArray], NPArray]],
    degree_raise: int = 3,
) -> NPArray:
    # See https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html
    # Create higher order function space (for reliable error norm computation)
    element = u_approx.function_space.ufl_element()
    raised_element = element.reconstruct(degree=element.degree() + degree_raise)
    mesh = u_approx.function_space.mesh
    func_space_raised = fem.FunctionSpace(mesh, raised_element)

    # Interpolate approximate solution
    u_approx_raised = fem.Function(func_space_raised)
    u_approx_raised.interpolate(u_approx)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_exact_raised = fem.Function(func_space_raised)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expression = fem.Expression(
            u_exact, func_space_raised.element.interpolation_points()
        )
        u_exact_raised.interpolate(u_expression)
    else:
        u_exact_raised.interpolate(u_exact)

    # Compute the error in the higher order function space
    error_raised = fem.Function(func_space_raised)
    error_raised.x.array[:] = u_approx_raised.x.array - u_exact_raised.x.array

    # Integrate the error
    error = fem.form(ufl.inner(error_raised, error_raised) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


reference_solution = calculate_approximate_solution(fem_mesh_resolution_reference)
u_exact = reference_solution.function


for mesh_resolution in fem_mesh_resolution_tests:
    approximate_solution = calculate_approximate_solution(mesh_resolution)
    u_approx = approximate_solution.function
    l2_error = calculate_l2_error(u_approx, u_exact)
    num_dofs = approximate_solution.coordinates_x.size
    print(f"Mesh: resolution: {mesh_resolution} \t L2 error: {l2_error} \t Number DOFs: {num_dofs}")