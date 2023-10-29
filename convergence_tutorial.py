from datetime import date
from typing import Callable, TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ufl
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from mpi4py import MPI
from scipy import stats
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

from parametricpinn.fem.base import DFunction, UFLExpression
from parametricpinn.fem.convergence import (
    calculate_infinity_error,
    calculate_l2_error,
    calculate_relative_l2_error,
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
inifinity_error_record: list[float] = []
num_dofs_record: list[int] = []
num_total_elements_record: list[int] = []


print("Start convergence analysis")
for num_elements in num_elements_tests:
    u_approx = solve_poisson(num_elements)
    l2_error = calculate_l2_error(u_approx, u_exact).item()
    relative_l2_error = calculate_relative_l2_error(u_approx, u_exact).item()
    infinity_error = calculate_infinity_error(u_approx, u_exact).item()
    num_total_elements = u_approx.function_space.mesh.topology.index_map(2).size_global
    num_dofs = u_approx.function_space.tabulate_dof_coordinates().size
    element_size_record.append(1 / num_elements)
    l2_error_record.append(l2_error)
    relative_l2_error_record.append(relative_l2_error)
    inifinity_error_record.append(infinity_error)
    num_total_elements_record.append(num_total_elements)
    num_dofs_record.append(num_dofs)

# Save results
results_frame = pd.DataFrame(
    {
        "element_size": element_size_record,
        "l2 error": l2_error_record,
        "relative l2 error": relative_l2_error_record,
        "infinity error": inifinity_error_record,
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


# Plot log l2 error
figure, axes = plt.subplots()
log_element_size = np.log(np.array(element_size_record))
log_l2_error = np.log(np.array(l2_error_record))

slope, intercept, _, _, _ = stats.linregress(log_element_size, log_l2_error)
regression_log_l2_error = slope * log_element_size + intercept

axes.plot(log_element_size, log_l2_error, "ob", label="simulation")
axes.plot(log_element_size, regression_log_l2_error, "--r", label="regression")
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
regression_log_infinity_error = slope * log_element_size + intercept

axes.plot(log_element_size, log_infinity_error, "ob", label="simulation")
axes.plot(log_element_size, regression_log_infinity_error, "--r", label="regression")
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
