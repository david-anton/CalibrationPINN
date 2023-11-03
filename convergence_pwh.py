from datetime import date
from typing import Callable, TypeAlias, Union

import dolfinx
import gmsh
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import Constant, dirichletbc, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import create_rectangle, locate_entities_boundary, meshtags
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpi4py import MPI
from scipy.interpolate import griddata

from parametricpinn.fem.base import (
    DFunction,
    DMesh,
    UFLExpression,
    UFLOperator,
    UFLTrialFunction,
)
from parametricpinn.fem.convergenceanalysis import (
    calculate_empirical_convegrence_order,
    infinity_error,
    l2_error,
    plot_error_convergence_analysis,
    relative_l2_error,
)
from parametricpinn.fem.utility import evaluate_function
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.settings import Settings, set_default_dtype
from parametricpinn.types import NPArray, PLTAxes

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
radius = 10.0
youngs_modulus = 210000.0
poissons_ratio = 0.3
traction_left_x = -100.0
traction_left_y = 0.0
volume_force_x = 0.0
volume_force_y = 0.0
is_symmetry_bc = True
is_hole = True
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_cell_type = dolfinx.mesh.CellType.triangle  # dolfinx.mesh.CellType.quadrilateral
fem_num_elements_reference = 128
fem_minimal_num_elements = 16
fem_reduction_factor = 1 / 2
fem_num_elements_tests = (
    np.array(
        [1, 1 / fem_reduction_factor, (1 / fem_reduction_factor) ** 2], dtype=np.int32
    )
    * fem_minimal_num_elements
).tolist()
fem_degree_raise = 3
# Plot
interpolation_method = "nearest"
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
    if is_hole:
        geometric_dim = 2
        mpi_rank = 0
        solid_marker = 1

        gmsh.initialize()
        geometry_kernel = gmsh.model.occ
        gmsh.model.add("domain")
        plate = geometry_kernel.add_rectangle(0, 0, 0, -edge_length, edge_length)
        hole = geometry_kernel.add_disk(0, 0, 0, radius, radius)
        geometry = geometry_kernel.cut([(2, plate)], [(2, hole)])

        geometry_kernel.synchronize()
        surface = geometry_kernel.getEntities(dim=2)
        assert surface == geometry[0]
        gmsh.model.addPhysicalGroup(surface[0][0], [surface[0][1]], solid_marker)
        gmsh.model.setPhysicalName(surface[0][0], solid_marker, "Solid")

        element_size = edge_length / num_elements
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)

        geometry_kernel.synchronize()
        gmsh.model.mesh.generate(geometric_dim)
        gmesh = gmsh.model
        mesh, _, _ = model_to_mesh(gmesh, communicator, mpi_rank, gdim=geometric_dim)
        gmsh.finalize()
    else:
        mesh = create_rectangle(
            comm=communicator,
            points=[[-edge_length, 0.0], [0.0, edge_length]],
            n=[num_elements, num_elements],
            cell_type=fem_cell_type,
        )

    # Function space
    shape = (2,)
    element = (fem_element_family, fem_element_degree, shape)
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

    tag_left = 0
    tag_right = 1
    tag_top = 2
    tag_bottom = 3
    bc_facets_dim = mesh.topology.dim - 1
    locate_left_facet = lambda x: np.isclose(x[0], -edge_length)
    locate_right_facet = lambda x: np.isclose(x[0], 0.0)
    locate_top_facet = lambda x: np.isclose(x[1], edge_length)
    locate_bottom_facet = lambda x: np.isclose(x[1], 0.0)
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
    dx = ufl.Measure("dx", domain=mesh)  # ufl.dx

    # mu_ = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    # lambda_ = (
    #     youngs_modulus
    #     * poissons_ratio
    #     / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    # )
    # # plane-stress
    # lambda_ = 2 * mu_ * lambda_ / (lambda_ + 2 * mu_)

    # def epsilon(u):
    #     return ufl.sym(ufl.grad(u))

    # def sigma(u):
    #     return 2.0 * mu_ * epsilon(u) + lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(2)

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
    rhs = ufl.dot(volume_force, test_function) * dx

    # Boundary conditions application
    T_left = Constant(
        mesh,
        default_scalar_type((traction_left_x, traction_left_y)),
    )
    bc_N_T_left = ufl.dot(T_left, test_function) * ds(tag_left)
    rhs = rhs + bc_N_T_left

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
            method=interpolation_method,
        )

    def cut_result_grid(result_grid: NPArray) -> NPArray:
        return result_grid[1:, 1:]

    def configure_ticks(axes: PLTAxes) -> None:
        x_min = np.nanmin(coordinates_x)
        x_max = np.nanmax(coordinates_x)
        y_min = np.nanmin(coordinates_y)
        y_max = np.nanmax(coordinates_y)
        x_ticks = np.linspace(x_min, x_max, num=3, endpoint=True)
        y_ticks = np.linspace(y_min, y_max, num=3, endpoint=True)
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(map(str, x_ticks.round(decimals=2)))
        axes.set_yticks(y_ticks)
        axes.set_yticklabels(map(str, y_ticks.round(decimals=2)))
        axes.tick_params(axis="both", which="major", pad=15)

    def add_geometry_specific_patches(axes: PLTAxes) -> None:
        def add_quarter_hole(axes: PLTAxes):
            hole = plt.Circle(
                (0.0, 0.0),
                radius=radius,
                color="white",
            )
            axes.add_patch(hole)

        if is_hole:
            add_quarter_hole(axes)

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
    configure_ticks(axes)
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
    add_geometry_specific_patches(axes)
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
    configure_ticks(axes)
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
    add_geometry_specific_patches(axes)
    output_path = project_directory.create_output_file_path(
        file_name, output_subdirectory
    )
    figure.savefig(output_path, bbox_inches="tight", dpi=256)
    plt.clf()


u_exact = calculate_approximate_solution(num_elements=fem_num_elements_reference)
plot_solution(u_exact, fem_num_elements_reference)

element_size_record: list[float] = []
l2_error_record: list[float] = []
relative_l2_error_record: list[float] = []
infinity_error_record: list[float] = []
num_dofs_record: list[int] = []
num_total_elements_record: list[int] = []
displacement_x_upper_left_corner_record: list[int] = []


print("Start convergence analysis")
for num_elements in fem_num_elements_tests:
    u_approx = calculate_approximate_solution(num_elements)
    u_approx_x = u_approx.sub(0).collapse()
    u_approx_y = u_approx.sub(1).collapse()
    plot_solution(u_approx, num_elements)
    num_total_elements = u_approx.function_space.mesh.topology.index_map(2).size_global
    num_dofs = u_approx.function_space.tabulate_dof_coordinates().size
    element_size_record.append(edge_length / num_elements)
    l2_error_record.append(l2_error(u_approx, u_exact))
    relative_l2_error_record.append(relative_l2_error(u_approx, u_exact))
    infinity_error_record.append(infinity_error(u_approx, u_exact))
    num_total_elements_record.append(num_total_elements)
    num_dofs_record.append(num_dofs)
    displacement_x_upper_left_corner = evaluate_function(
        u_approx_x, np.array([[-edge_length, edge_length, 0.0]])
    )[0]
    displacement_x_upper_left_corner_record.append(displacement_x_upper_left_corner)

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

convergence_x_upper_left_corner = calculate_empirical_convegrence_order(
    displacement_x_upper_left_corner_record, fem_reduction_factor
)
print(f"Convergence u_x upper left corner: {convergence_x_upper_left_corner}")

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
