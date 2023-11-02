from datetime import date
from typing import Callable, TypeAlias, Union

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from scipy import stats

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


FacetsDim: TypeAlias = int
LocateFacetFunc: TypeAlias = Callable[[NPArray], npt.NDArray[np.bool_]]
FacetTag: TypeAlias = int
Boundary: TypeAlias = tuple[FacetTag, LocateFacetFunc]
BoundaryList: TypeAlias = list[Boundary]
SortedFacetIndices: TypeAlias = npt.NDArray[np.int32]
SortedFacetTags: TypeAlias = npt.NDArray[np.int32]

UExact: TypeAlias = Union[DFunction, UFLExpression, Callable[[NPArray], NPArray]]

# Setup
length = 1
width = 0.2
mu_ = 1
rho = 1
delta = width / length
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma
# FEM
element_family = "Lagrange"
element_degree = 1
cell_type = dolfinx.mesh.CellType.hexahedron
num_elements_reference = 64
num_elements_tests = [4, 8, 16, 32]
degree_raise = 3
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdirectory = f"{current_date}_convergence_beam"


communicator = MPI.COMM_WORLD


def calculate_approximate_solution(num_elements):
    print(f"Run simulation with {num_elements} elements per edge")
    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([length, width, width])],
        [5 * num_elements, num_elements, num_elements],
        cell_type=cell_type,
    )
    V = fem.VectorFunctionSpace(domain, (element_family, element_degree))

    def clamped_boundary(x):
        return np.isclose(x[0], 0)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    T = fem.Constant(domain, default_scalar_type((0, 0, 0)))

    ds = ufl.Measure("ds", domain=domain)

    def epsilon(u):
        return ufl.sym(
            ufl.grad(u)
        )  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu_ * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    file_name = f"deformation_{num_elements}.xdmf"
    output_path = project_directory.create_output_file_path(
        file_name, output_subdirectory
    )
    with dolfinx.io.XDMFFile(domain.comm, output_path, "w") as xdmf:
        xdmf.write_mesh(domain)
        uh.name = "Deformation"
        xdmf.write_function(uh)
    return uh


u_exact = calculate_approximate_solution(num_elements=num_elements_reference)

element_size_record: list[float] = []
l2_error_record: list[float] = []
relative_l2_error_record: list[float] = []
inifinity_error_record: list[float] = []
num_dofs_record: list[int] = []
num_total_elements_record: list[int] = []


print("Start convergence analysis")
for num_elements in num_elements_tests:
    u_approx = calculate_approximate_solution(num_elements)
    l2_error = calculate_l2_error(u_approx, u_exact).item()
    relative_l2_error = calculate_relative_l2_error(u_approx, u_exact).item()
    infinity_error = calculate_infinity_error(u_approx, u_exact).item()
    num_total_elements = u_approx.function_space.mesh.topology.index_map(2).size_global
    num_dofs = u_approx.function_space.tabulate_dof_coordinates().size
    element_size_record.append(
        length / num_elements
    )  # element_size_record.append(edge_length / num_elements)
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

# Convergence rate
error_k_1 = l2_error_record[-3]
error_k_2 = l2_error_record[-2]
error_k_4 = l2_error_record[-1]
convergence_rate = (1 / np.log(2)) * np.log(
    np.absolute((error_k_1 - error_k_2) / (error_k_2 - error_k_4))
)
print(f"Convergence rate: {convergence_rate}")

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
