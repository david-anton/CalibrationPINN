from datetime import date
from typing import Callable, TypeAlias, Union

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, dirichletbc, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle, locate_entities_boundary, meshtags
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpi4py import MPI
from scipy import stats
from scipy.interpolate import griddata

from parametricpinn.fem.base import (
    DFunction,
    DMesh,
    UFLExpression,
    UFLOperator,
    UFLTrialFunction,
)
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
edge_length = 100.0
youngs_modulus = 210000.0
poissons_ratio = 0.3
volume_force_x = 0.0
volume_force_y = 0.0
traction_left_x = -100.0
traction_left_y = 0.0
is_symmetry_bc = False
# FEM
element_family = "Lagrange"
element_degree = 1
cell_type = dolfinx.mesh.CellType.triangle  # dolfinx.mesh.CellType.quadrilateral
num_elements_reference = 128
num_elements_tests = [1, 2, 4, 8, 16, 32, 64]
degree_raise = 3
# Plot
color_map = "jet"
num_points_per_edge = 256
ticks_max_number_of_intervals = 255
num_cbar_ticks = 7
plot_font_size = 12
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdirectory = f"{current_date}_convergence_pwh"


communicator = MPI.COMM_WORLD


def calculate_approximate_solution(num_elements):
    print(f"Run simulation with {num_elements} elements per edge")
    # Mesh
    mesh = create_rectangle(
        comm=communicator,
        points=[[-edge_length, 0.0], [0.0, edge_length]],
        n=[num_elements, num_elements],
        cell_type=cell_type,
    )

    # Function space
    shape = (2,)
    element = (element_family, element_degree, shape)
    function_space = functionspace(mesh, element)
    trial_function = ufl.TrialFunction(function_space)
    test_function = ufl.TestFunction(function_space)

    # Boundary condition definition
    def list_sorted_facet_indices_and_tags(
        boundaries: BoundaryList, mesh: DMesh, bc_facets_dim: FacetsDim
    ) -> tuple[SortedFacetIndices, SortedFacetTags]:
        facet_indices_list: list[npt.NDArray[np.int32]] = []
        facet_tags_list: list[npt.NDArray[np.int32]] = []
        for tag, locator_func in boundaries:
            located_facet_indices = locate_entities_boundary(
                mesh, bc_facets_dim, locator_func
            )
            facet_indices_list.append(located_facet_indices)
            facet_tags_list.append(np.full_like(located_facet_indices, tag))
        facet_indices = np.hstack(facet_indices_list).astype(np.int32)
        facet_tags = np.hstack(facet_tags_list).astype(np.int32)
        sorting_index_array = np.argsort(facet_indices)
        sorted_facet_indices = facet_indices[sorting_index_array]
        sorted_facet_tags = facet_tags[sorting_index_array]
        return sorted_facet_indices, sorted_facet_tags

    tag_right = 0
    tag_left = 1
    tag_bottom = 2
    tag_top = 3
    bc_facets_dim = mesh.topology.dim - 1
    locate_right_facet = lambda x: np.isclose(x[0], 0.0)
    locate_left_facet = lambda x: np.isclose(x[0], -edge_length)
    locate_bottom_facet = lambda x: np.isclose(x[1], 0.0)
    locate_top_facet = lambda x: np.isclose(x[1], edge_length)
    boundaries = [
        (tag_right, locate_right_facet),
        (tag_left, locate_left_facet),
        (tag_bottom, locate_bottom_facet),
        (tag_top, locate_top_facet),
    ]
    sorted_facet_indices, sorted_facet_tags = list_sorted_facet_indices_and_tags(
        boundaries=boundaries, mesh=mesh, bc_facets_dim=bc_facets_dim
    )
    boundary_tags = meshtags(
        mesh,
        bc_facets_dim,
        sorted_facet_indices,
        sorted_facet_tags,
    )

    # Variational form
    volume_force = Constant(
        mesh, (default_scalar_type((volume_force_x, volume_force_y)))
    )

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundary_tags)
    dx = ufl.dx  # ufl.Measure("dx", domain=mesh)

    # mu_ = youngs_modulus / (2 * (1 + poissons_ratio))
    # lambda_ = (
    #     youngs_modulus
    #     * poissons_ratio
    #     / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
    # )
    # lambda_ = 2 * mu_ * lambda_ / (lambda_ + 2 * mu_)

    # def sigma(u: UFLTrialFunction) -> UFLOperator:
    #     # return lambda_*ufl.tr(epsilon(u))*ufl.Identity(2) + 2.0*mu_*epsilon(u)
    #     return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu_ * epsilon(u)

    # def epsilon(u: UFLTrialFunction) -> UFLOperator:
    #     return ufl.sym(ufl.grad(u))

    compliance_matrix = (1 / youngs_modulus) * ufl.as_matrix(
        [
            [1.0, -poissons_ratio, 0.0],
            [-poissons_ratio, 1.0, 0.0],
            [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
        ]
    )
    elasticity_matrix = ufl.inv(compliance_matrix)

    def sigma(u: UFLTrialFunction) -> UFLOperator:
        return _sigma_voigt_to_matrix(
            ufl.dot(elasticity_matrix, _epsilon_matrix_to_voigt(epsilon(u)))
        )

    def _epsilon_matrix_to_voigt(eps: UFLOperator) -> UFLOperator:
        return ufl.as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

    def _sigma_voigt_to_matrix(sig: UFLOperator) -> UFLOperator:
        return ufl.as_tensor([[sig[0], sig[2]], [sig[2], sig[1]]])

    def epsilon(u: UFLTrialFunction) -> UFLOperator:
        return ufl.sym(ufl.grad(u))

    lhs = ufl.inner(sigma(trial_function), epsilon(test_function)) * dx
    rhs = ufl.dot(test_function, volume_force) * dx

    # Boundary conditions application
    T_left = Constant(
        mesh,
        default_scalar_type((traction_left_x, traction_left_y)),
    )
    bc_N_T_left = ufl.dot(T_left, test_function) * ds(tag_left)
    rhs = rhs + bc_N_T_left

    # T_top = Constant(
    #     mesh,
    #     default_scalar_type((0.0, 0.0)),
    # )
    # bc_N_T_top = ufl.dot(T_top, test_function) * ds(tag_top)
    # rhs = rhs + bc_N_T_top

    if is_symmetry_bc:
        displacement_right_x = 0.0
        displacement_bottom_y = 0.0
        u_right_x = Constant(
            mesh,
            default_scalar_type(displacement_right_x),
        )
        u_bottom_y = Constant(
            mesh,
            default_scalar_type(displacement_bottom_y),
        )

        facets_u_right = boundary_tags.find(tag_right)
        dofs_u_right_x = locate_dofs_topological(
            function_space.sub(0), bc_facets_dim, facets_u_right
        )
        bc_D_u_right_x = dirichletbc(u_right_x, dofs_u_right_x, function_space.sub(0))

        facets_u_bottom = boundary_tags.find(tag_bottom)
        dofs_u_bottom_y = locate_dofs_topological(
            function_space.sub(1), bc_facets_dim, facets_u_bottom
        )
        bc_D_u_bottom_y = dirichletbc(
            u_bottom_y, dofs_u_bottom_y, function_space.sub(1)
        )

        dirichlet_bcs = [bc_D_u_right_x, bc_D_u_bottom_y]

    else:
        displacement_right_x = 0.0
        displacement_right_y = 0.0

        u_right = Constant(
            mesh,
            default_scalar_type((displacement_right_x, displacement_right_y)),
        )

        facets_u_right = boundary_tags.find(tag_right)
        dofs_u_right = locate_dofs_topological(
            function_space, bc_facets_dim, facets_u_right
        )
        bc_D_u_right = dirichletbc(u_right, dofs_u_right, function_space)

        dirichlet_bcs = [bc_D_u_right]

    # Problem
    problem = LinearProblem(
        lhs,
        rhs,
        bcs=dirichlet_bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    approximate_solution = problem.solve()

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    file_name = f"facet_tags_{num_elements}.xdmf"
    output_path = project_directory.create_output_file_path(
        file_name, output_subdirectory
    )
    with XDMFFile(
        mesh.comm, output_path, "w", encoding=XDMFFile.Encoding.ASCII
    ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(boundary_tags, mesh.geometry)
        xdmf.write_function(approximate_solution)
    return approximate_solution


def plot_solution(function: DFunction, num_elements: int) -> None:
    def generate_coordinate_grid(num_points_per_edge: int) -> list[NPArray]:
        grid_coordinates_x = np.linspace(
            -edge_length,
            0.0,
            num=num_points_per_edge,
        )
        grid_coordinates_y = np.linspace(
            0.0,
            edge_length,
            num=num_points_per_edge,
        )
        return np.meshgrid(grid_coordinates_x, grid_coordinates_y)

    def interpolate_results_on_grid(
        results: NPArray,
        coordinates_x: NPArray,
        coordinates_y: NPArray,
        coordinates_grid_x: NPArray,
        coordinates_grid_y: NPArray,
    ) -> NPArray:
        results = results.reshape((-1,))
        coordinates = np.concatenate((coordinates_x, coordinates_y), axis=1)
        return griddata(
            coordinates,
            results,
            (coordinates_grid_x, coordinates_grid_y),
            method="cubic",
        )

    def cut_result_grid(result_grid: NPArray) -> NPArray:
        return result_grid[1:, 1:]

    # Extract FEM results
    coordinates = function.function_space.tabulate_dof_coordinates()
    coordinates_x = coordinates[:, 0].reshape((-1, 1))
    coordinates_y = coordinates[:, 1].reshape((-1, 1))

    displacements = function.x.array.reshape(
        (-1, function.function_space.mesh.geometry.dim)
    )
    displacements_x = displacements[:, 0].reshape((-1, 1))
    displacements_y = displacements[:, 1].reshape((-1, 1))

    # Interpolate results on grid
    coordinates_grid_x, coordinates_grid_y = generate_coordinate_grid(
        num_points_per_edge
    )
    displacements_grid_x = interpolate_results_on_grid(
        displacements_x,
        coordinates_x,
        coordinates_y,
        coordinates_grid_x,
        coordinates_grid_y,
    )
    displacements_grid_y = interpolate_results_on_grid(
        displacements_y,
        coordinates_x,
        coordinates_y,
        coordinates_grid_x,
        coordinates_grid_y,
    )

    # Plot interpolated results
    results = displacements_grid_x
    title = "Displacements x"
    file_name = f"displacement_field_x_{num_elements}.png"
    figure, axes = plt.subplots()
    axes.set_title(title)
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    min_value = np.nanmin(results)
    max_value = np.nanmax(results)
    tick_values = MaxNLocator(nbins=ticks_max_number_of_intervals).tick_values(
        min_value, max_value
    )
    normalizer = BoundaryNorm(tick_values, ncolors=plt.get_cmap(color_map).N, clip=True)
    plot = axes.pcolormesh(
        coordinates_grid_x, coordinates_grid_y, results, cmap=color_map, norm=normalizer
    )
    cbar_ticks = (
        np.linspace(min_value, max_value, num=num_cbar_ticks, endpoint=True)
        .round(decimals=4)
        .tolist()
    )
    cbar = figure.colorbar(mappable=plot, ax=axes, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(map(str, cbar_ticks))
    cbar.ax.minorticks_off()
    figure.axes[1].tick_params(labelsize=plot_font_size)
    output_path = project_directory.create_output_file_path(
        file_name, output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=256)
    plt.clf()

    results = displacements_grid_y
    title = "Displacements y"
    file_name = f"displacement_field_y_{num_elements}.png"
    figure, axes = plt.subplots()
    axes.set_title(title)
    axes.set_xlabel("x [mm]")
    axes.set_ylabel("y [mm]")
    min_value = np.nanmin(results)
    max_value = np.nanmax(results)
    tick_values = MaxNLocator(nbins=ticks_max_number_of_intervals).tick_values(
        min_value, max_value
    )
    normalizer = BoundaryNorm(tick_values, ncolors=plt.get_cmap(color_map).N, clip=True)
    plot = axes.pcolormesh(
        coordinates_grid_x, coordinates_grid_y, results, cmap=color_map, norm=normalizer
    )
    cbar_ticks = (
        np.linspace(min_value, max_value, num=num_cbar_ticks, endpoint=True)
        .round(decimals=4)
        .tolist()
    )
    cbar = figure.colorbar(mappable=plot, ax=axes, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(map(str, cbar_ticks))
    cbar.ax.minorticks_off()
    figure.axes[1].tick_params(labelsize=plot_font_size)
    output_path = project_directory.create_output_file_path(
        file_name, output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=256)
    plt.clf()


u_exact = calculate_approximate_solution(num_elements=num_elements_reference)
plot_solution(u_exact, num_elements_reference)

element_size_record: list[float] = []
l2_error_record: list[float] = []
relative_l2_error_record: list[float] = []
inifinity_error_record: list[float] = []
num_dofs_record: list[int] = []
num_total_elements_record: list[int] = []


print("Start convergence analysis")
for num_elements in num_elements_tests:
    u_approx = calculate_approximate_solution(num_elements)
    plot_solution(u_approx, num_elements)
    l2_error = calculate_l2_error(u_approx, u_exact).item()
    relative_l2_error = calculate_relative_l2_error(u_approx, u_exact).item()
    infinity_error = calculate_infinity_error(u_approx, u_exact).item()
    num_total_elements = u_approx.function_space.mesh.topology.index_map(2).size_global
    num_dofs = u_approx.function_space.tabulate_dof_coordinates().size
    element_size_record.append(edge_length / num_elements)
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
