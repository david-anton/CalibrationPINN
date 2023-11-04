from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

from parametricpinn.fem import (
    NeoHookeanProblemConfig,
    QuarterPlateWithHoleDomainConfig,
    SimulationConfig,
    run_simulation,
)
from parametricpinn.fem.base import DFunction
from parametricpinn.fem.convergenceanalysis import (
    calculate_empirical_convegrence_order,
    h01_error,
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

### Configuration
# Set up
edge_length = 100.0
radius = 10.0
traction_left_x = 0.0
traction_left_y = 0.0
volume_force_x = -1.0
volume_force_y = 0.0
youngs_modulus = 2000.0
poissons_ratio = 0.3
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 2
fem_element_size_reference = 0.2
fem_element_size_largest = 1.6
fem_reduction_factor = 1 / 2
fem_element_size_tests = (
    np.array([1, fem_reduction_factor, fem_reduction_factor**2])
    * fem_element_size_largest
).tolist()
# Plot
num_points_per_edge = 256
interpolation_method = "nearest"
# Output
current_date = date.today().strftime("%Y%m%d")
output_subdirectory = (
    f"{current_date}_convergence_analysis_neohookean_quarterplatewithhole_f_only"
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


def plot_solution(function: DFunction, element_size: int) -> None:
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
    def plot_one_result_grid(
        result_grid: NPArray,
        dimension: str,
        coordinates_grid_x: NPArray,
        coordinates_grid_y: NPArray,
        element_size: float,
    ) -> None:
        ticks_max_number_of_intervals = 255
        color_map = "jet"
        num_cbar_ticks = 7
        font_size = 12

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

        def add_quarter_hole(axes: PLTAxes):
            hole = plt.Circle(
                (0.0, 0.0),
                radius=radius,
                color="white",
            )
            axes.add_patch(hole)

        title = f"Displacements {dimension}"
        file_name = f"displacement_field_{dimension}_{element_size}.png"
        figure, axes = plt.subplots()
        axes.set_title(title)
        axes.set_xlabel("x [mm]")
        axes.set_ylabel("y [mm]")
        configure_ticks(axes)
        min_value = np.nanmin(result_grid)
        max_value = np.nanmax(result_grid)
        tick_values = MaxNLocator(nbins=ticks_max_number_of_intervals).tick_values(
            min_value, max_value
        )
        normalizer = BoundaryNorm(
            tick_values, ncolors=plt.get_cmap(color_map).N, clip=True
        )
        plot = axes.pcolormesh(
            coordinates_grid_x,
            coordinates_grid_y,
            result_grid,
            cmap=color_map,
            norm=normalizer,
        )
        cbar_ticks = (
            np.linspace(min_value, max_value, num=num_cbar_ticks, endpoint=True)
            .round(decimals=4)
            .tolist()
        )
        cbar = figure.colorbar(mappable=plot, ax=axes, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(map(str, cbar_ticks))
        cbar.ax.minorticks_off()
        figure.axes[1].tick_params(labelsize=font_size)
        add_quarter_hole(axes)
        output_path = project_directory.create_output_file_path(
            file_name, output_subdirectory
        )
        figure.savefig(output_path, bbox_inches="tight", dpi=256)
        plt.clf()

    plot_one_result_grid(
        result_grid=displacements_grid_x,
        dimension="x",
        coordinates_grid_x=coordinates_grid_x,
        coordinates_grid_y=coordinates_grid_y,
        element_size=element_size,
    )
    plot_one_result_grid(
        result_grid=displacements_grid_y,
        dimension="y",
        coordinates_grid_x=coordinates_grid_x,
        coordinates_grid_y=coordinates_grid_y,
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
    function = results.function
    plot_solution(function, element_size)
    return function


u_exact = calculate_approximate_solution(fem_element_size_reference)
u_exact_x = u_exact.sub(0).collapse()
u_exact_y = u_exact.sub(1).collapse()


element_size_record: list[float] = []
l2_error_record: list[float] = []
relative_l2_error_record: list[float] = []
h01_error_x_record: list[float] = []
h01_error_y_record: list[float] = []
infinity_error_record: list[float] = []
num_dofs_record: list[int] = []
num_elements_record: list[int] = []
displacement_x_beside_hole_record: list[float] = []
displacement_y_above_hole_record: list[float] = []
displacement_x_upper_left_corner_record: list[float] = []
displacement_y_upper_left_corner_record: list[float] = []


print("Start convergence analysis")
for element_size in fem_element_size_tests:
    u_approx = calculate_approximate_solution(element_size)
    u_approx_x = u_approx.sub(0).collapse()
    u_approx_y = u_approx.sub(1).collapse()
    num_elements = u_approx.function_space.mesh.topology.index_map(2).size_global
    num_dofs = u_approx.function_space.tabulate_dof_coordinates().size
    l2_error_record.append(l2_error(u_approx, u_exact))
    relative_l2_error_record.append(relative_l2_error(u_approx, u_exact))
    h01_error_x_record.append(h01_error(u_approx_x, u_exact_x))
    h01_error_y_record.append(h01_error(u_approx_y, u_exact_y))
    infinity_error_record.append(infinity_error(u_approx, u_exact))
    num_elements_record.append(num_elements)
    num_dofs_record.append(num_dofs)
    element_size_record.append(element_size)
    displacement_x_beside_hole = evaluate_function(
        u_approx_x, np.array([[-radius, 0.0, 0.0]])
    )
    displacement_y_above_hole = evaluate_function(
        u_approx_y, np.array([[0.0, radius, 0.0]])
    )
    displacement_x_upper_left_corner = evaluate_function(
        u_approx_x, np.array([[-edge_length, edge_length, 0.0]])
    )
    displacement_y_upper_left_corner = evaluate_function(
        u_approx_y, np.array([[-edge_length, edge_length, 0.0]])
    )
    displacement_x_beside_hole_record.append(displacement_x_beside_hole)
    displacement_y_above_hole_record.append(displacement_y_above_hole)
    displacement_x_upper_left_corner_record.append(displacement_x_upper_left_corner)
    displacement_y_upper_left_corner_record.append(displacement_y_upper_left_corner)


# Save results
records_frame = pd.DataFrame(
    {
        "element_size": element_size_record,
        "L2 error": l2_error_record,
        "relative L2 error": relative_l2_error_record,
        "H01 error x": h01_error_x_record,
        "H01 error y": h01_error_y_record,
        "infinity error": infinity_error_record,
        "number elements": num_elements_record,
        "number dofs": num_dofs_record,
    }
)
pandas_data_writer = PandasDataWriter(project_directory)
pandas_data_writer.write(
    data=records_frame,
    file_name="error_norms",
    subdir_name=output_subdirectory,
    header=True,
)


convergence_x_beside_hole = calculate_empirical_convegrence_order(
    displacement_x_beside_hole_record, fem_reduction_factor
)

convergence_y_above_hole = calculate_empirical_convegrence_order(
    displacement_y_above_hole_record, fem_reduction_factor
)

convergence_x_upper_left_corner = calculate_empirical_convegrence_order(
    displacement_x_upper_left_corner_record, fem_reduction_factor
)

convergence_y_upper_left_corner = calculate_empirical_convegrence_order(
    displacement_y_upper_left_corner_record, fem_reduction_factor
)

convergence_frame = pd.DataFrame(
    {
        "u_x beside hole": convergence_x_beside_hole,
        "u_y above hole": convergence_y_above_hole,
        "u_x upper left corner": convergence_x_upper_left_corner,
        "u_y upper left corner": convergence_y_upper_left_corner,
    }
)
pandas_data_writer.write(
    data=convergence_frame,
    file_name="convergence_orders",
    subdir_name=output_subdirectory,
    header=True,
)


# Postprocessing
print("Postprocessing")

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

plot_error_convergence_analysis(
    error_record=h01_error_x_record,
    element_size_record=element_size_record,
    error_norm="h01_x",
    output_subdirectory=output_subdirectory,
    project_directory=project_directory,
)

plot_error_convergence_analysis(
    error_record=h01_error_y_record,
    element_size_record=element_size_record,
    error_norm="h01_y",
    output_subdirectory=output_subdirectory,
    project_directory=project_directory,
)
