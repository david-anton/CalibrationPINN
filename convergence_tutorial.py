from datetime import date
from typing import Callable, TypeAlias, Union

import numpy as np
import pandas as pd
import torch
import ufl
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

from parametricpinn.fem.base import DFunction, UFLExpression
from parametricpinn.fem.convergenceanalysis import (
    infinity_error,
    l2_error,
    plot_error_convergence_analysis,
    relative_l2_error,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.settings import Settings, set_default_dtype
from parametricpinn.types import NPArray

# Set up simulation
settings = Settings()
project_directory = ProjectDirectory(settings)
set_default_dtype(torch.float64)

UExact: TypeAlias = Union[DFunction, UFLExpression, Callable[[NPArray], NPArray]]


# FEM
element_family = "Lagrange"
element_degree = 1
num_elements_reference = 128
num_elements_tests = [4, 8, 16, 32, 64]
degree_raise = 3
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdirectory = f"{current_date}_convergence_study_laplace_tutorial"


def u_ex(mod):
    return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])


u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

communicator = MPI.COMM_WORLD


def solve_poisson(num_elements):
    mesh = create_unit_square(communicator, num_elements, num_elements)
    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ufl(x)))
    V = functionspace(mesh, (element_family, element_degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx
    u_bc = Function(V)
    u_bc.interpolate(u_numpy)
    facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True)
    )
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [dirichletbc(u_bc, dofs)]
    default_problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    return default_problem.solve()


u_exact = solve_poisson(num_elements=num_elements_reference)

element_size_record: list[float] = []
l2_error_record: list[float] = []
relative_l2_error_record: list[float] = []
infinity_error_record: list[float] = []
num_dofs_record: list[int] = []
num_total_elements_record: list[int] = []


print("Start convergence analysis")
for num_elements in num_elements_tests:
    u_approx = solve_poisson(num_elements)
    num_total_elements = u_approx.function_space.mesh.topology.index_map(2).size_global
    num_dofs = u_approx.function_space.tabulate_dof_coordinates().size
    element_size_record.append(1 / num_elements)
    l2_error_record.append(l2_error(u_approx, u_exact))
    relative_l2_error_record.append(relative_l2_error(u_approx, u_exact))
    infinity_error_record.append(infinity_error(u_approx, u_exact))
    num_total_elements_record.append(num_total_elements)
    num_dofs_record.append(num_dofs)

# Save results
results_frame = pd.DataFrame(
    {
        "element_size": element_size_record,
        "l2 error": l2_error_record,
        "relative l2 error": relative_l2_error_record,
        "infinity error": infinity_error_record,
        "number elements": num_total_elements_record,
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
print("Postprocessing")

# Convergence rate
error_k_1 = l2_error_record[-3]
error_k_2 = l2_error_record[-2]
error_k_4 = l2_error_record[-1]
convergence_rate = (1 / np.log(2)) * np.log(
    np.absolute((error_k_1 - error_k_2) / (error_k_2 - error_k_4))
)
print(f"Convergence rate: {convergence_rate}")

plot_error_convergence_analysis(
    error_record=l2_error_record,
    element_size_record=element_size_record,
    error_norm="l2",
    output_subdirectory=output_subdirectory,
    project_directory=project_directory,
)

plot_error_convergence_analysis(
    error_record=infinity_error_record,
    element_size_record=element_size_record,
    error_norm="infinity",
    output_subdirectory=output_subdirectory,
    project_directory=project_directory,
)
