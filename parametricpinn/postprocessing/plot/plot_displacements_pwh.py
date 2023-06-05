from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

from parametricpinn.fem.platewithhole import run_simulation
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Module, NPArray, PLTFigure


class DisplacementsPlotterConfigPWH:
    def __init__(self) -> None:
        # font sizes
        # self.label_size = 20
        # font size in legend
        self.font_size = 16
        self.font = {"size": self.label_size}

        # title pad
        self.title_pad = 10

        # major ticks
        self.major_tick_label_size = 20
        self.major_ticks_size = self.font_size
        self.major_ticks_width = 2
        # minor ticks
        self.minor_tick_label_size = 14
        self.minor_ticks_size = 12
        self.minor_ticks_width = 1

        # scientific notation
        self.scientific_notation_size = self.font_size

        # color map
        self.color_map = "jet"

        # legend
        self.ticks_max_number_of_intervals = 255
        self.num_cbar_ticks = 7

        # resolution of results
        num_points_per_edge = 128

        # save options
        self.dpi = 300

        # labels
        self.x_label = "x [mm]"
        self.y_label = "y [mm]"


SimulationConfig = namedtuple(
    "SimulationConfig",
    [
        "youngs_modulus",
        "poissons_ratio",
        "model",
        "edge_length",
        "radius",
        "volume_force_x",
        "volume_force_y",
        "traction_left_x",
        "traction_left_y",
        "mesh_resolution",
    ],
)

SimulationResults = namedtuple(
    "SimulationResults",
    [
        "coordinates_x",
        "coordinates_y",
        "fem_displacements_x",
        "fem_displacements_y",
        "pinn_displacements_x",
        "pinn_displacements_y",
        "re_displacements_x",
        "re_displacements_y",
    ],
)


def plot_displacements_pwh(
    ansatz: Module,
    youngs_modulus_and_poissons_ratio: list[tuple[float, float]],
    model: str,
    edge_length: float,
    radius: float,
    volume_force_x: float,
    volume_force_y: float,
    traction_left_x: float,
    traction_left_y: float,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfigPWH,
    mesh_resolution=0.5,
) -> None:
    ansatz.eval()
    ansatz.cpu()
    for parameters in youngs_modulus_and_poissons_ratio:
        youngs_modulus = parameters[0]
        poissons_ratio = parameters[1]
        simulation_config = SimulationConfig(
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
            model=model,
            edge_length=edge_length,
            radius=radius,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            traction_left_x=traction_left_x,
            traction_left_y=traction_left_y,
            mesh_resolution=mesh_resolution,
        )
        _plot_one_simulation(
            ansatz, simulation_config, output_subdir, project_directory, plot_config
        )


def _plot_one_simulation(
    ansatz: Module,
    simulation_config: SimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfigPWH,
):
    simulation_results = _calculate_results(
        ansatz, simulation_config, output_subdir, project_directory
    )


def _calculate_results(
    ansatz: Module,
    simulation_config: SimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> SimulationResults:
    (
        coordinates_x,
        coordinates_y,
        fem_displacements_x,
        fem_displacements_y,
    ) = _run_simulation(simulation_config, output_subdir, project_directory)
    pinn_displacements_x, pinn_displacements_y = _run_prametric_pinn(
        ansatz, simulation_config, coordinates_x, coordinates_y
    )
    re_displacements_x = _calculate_relative_error(
        pinn_displacements_x, fem_displacements_x
    )
    re_displacements_y = _calculate_relative_error(
        pinn_displacements_y, fem_displacements_y
    )
    return SimulationResults(
        coordinates_x=coordinates_x,
        coordinates_y=coordinates_y,
        fem_displacements_x=fem_displacements_x,
        fem_displacements_y=fem_displacements_y,
        pinn_displacements_x=pinn_displacements_x,
        pinn_displacements_y=pinn_displacements_y,
        re_displacements_x=re_displacements_x,
        re_displacements_y=re_displacements_y,
    )


def _run_simulation(
    simulation_config: SimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> tuple[NPArray, NPArray, NPArray, NPArray]:
    results = run_simulation(
        model=simulation_config.model,
        youngs_modulus=simulation_config.youngs_modulus,
        poissons_ratio=simulation_config.poissons_ratio,
        edge_length=simulation_config.edge_length,
        radius=simulation_config.radius,
        volume_force_x=simulation_config.volume_force_x,
        volume_force_y=simulation_config.volume_force_y,
        traction_left_x=simulation_config.traction_left_x,
        traction_left_y=simulation_config.traction_left_y,
        save_results=False,
        save_metadata=False,
        output_subdir=output_subdir,
        project_directory=project_directory,
        mesh_resolution=simulation_config.mesh_resolution,
    )
    coordinates_x = results.coordinates_x
    coordinates_y = results.coordinates_y
    displacements_x = results.displacements_x
    displacements_y = results.displacements_y
    return coordinates_x, coordinates_y, displacements_x, displacements_y


def _run_prametric_pinn(
    ansatz: Module,
    simulation_config: SimulationConfig,
    coordinates_x: NPArray,
    coordinates_y: NPArray,
) -> tuple[NPArray, NPArray]:
    youngs_modulus = simulation_config.youngs_modulus
    poissons_ratio = simulation_config.poissons_ratio
    coordinates = torch.concat(
        (torch.from_numpy(coordinates_x), torch.from_numpy(coordinates_y)), dim=1
    )
    inputs = torch.concat(
        (
            coordinates,
            youngs_modulus.repeat(coordinates.size(dim=0), 1),
            poissons_ratio.repeat(coordinates.size(dim=0), 1),
        ),
        dim=1,
    )
    with torch.no_grad():
        outputs = ansatz(inputs)
    outputs = outputs.numpy()
    displacements_x = outputs[:, 0]
    displacements_y = outputs[:, 1]
    return displacements_x, displacements_y


def _calculate_relative_error(
    predicted_displacements: NPArray, simulated_displacements: NPArray
) -> NPArray:
    absolute_tolerance = 1e-7
    return (predicted_displacements - simulated_displacements) / (
        simulated_displacements + absolute_tolerance
    )


def _plot_results(
    simulation_results: SimulationResults,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfigPWH,
) -> None:
    pass


def _create_normalizer(
    results: NPArray, plot_config: DisplacementsPlotterConfigPWH
) -> BoundaryNorm:
    min_value = np.nanmin(results)
    max_value = np.nanmax(results)
    tick_values = MaxNLocator(
        nbins=plot_config.ticks_max_number_of_intervals
    ).tick_values(min_value, max_value)
    return BoundaryNorm(
        tick_values, ncolors=plt.get_cmap(plot_config.color_map).N, clip=True
    )


def _create_ticks(
    results: NPArray, plot_config: DisplacementsPlotterConfigPWH
) -> list[float]:
    min_value = np.nanmin(results)
    max_value = np.nanmax(results)
    ticks = (
        np.linspace(min_value, max_value, num=plot_config.num_cbar_ticks, endpoint=True)
        .round(decimals=4)
        .tolist()
    )
    return ticks


def _generate_coordinate_grid(
    coordinates_x: NPArray,
    coordinates_y: NPArray,
    plot_comfig: DisplacementsPlotterConfigPWH,
) -> list[NPArray]:
    grid_coordinates_x = np.linspace(
        np.amin(coordinates_x),
        np.amax(coordinates_x),
        num=plot_comfig.num_points_per_edge,
    )
    grid_coordinates_y = np.linspace(
        np.amin(coordinates_y),
        np.amax(coordinates_y),
        num=plot_comfig.num_points_per_edge,
    )
    return np.meshgrid(grid_coordinates_x, grid_coordinates_y)


def _interpolate_results_on_grid(
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
        method="linear",
    )


def _plot_once(
    results: NPArray,
    coordinates_x: NPArray,
    coordinates_y: NPArray,
    title: str,
    normalizer: BoundaryNorm,
    cbar_ticks: list[float],
    plot_config: DisplacementsPlotterConfigPWH,
) -> PLTFigure:
    figure, axes = plt.subplots()
    ################################################################################
    axes.set_title(title, pad=plot_config.title_pad, **plot_config.font)
    axes.set_xlabel(plot_config.x_label, **plot_config.font)
    axes.set_ylabel(plot_config.y_label, **plot_config.font)
    axes.tick_params(
        axis="both", which="minor", labelsize=plot_config.minor_tick_label_size
    )
    axes.tick_params(
        axis="both", which="major", labelsize=plot_config.major_tick_label_size
    )
    ################################################################################
    x_min = np.nanmin(coordinates_x)
    x_max = np.nanmax(coordinates_x)
    y_min = np.nanmin(coordinates_y)
    y_max = np.nanmax(coordinates_y)
    x_ticks = np.linspace(x_min, x_max, num=3, endpoint=True)
    y_ticks = np.linspace(y_min, y_max, num=3, endpoint=True)
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_xticks(x_ticks)
    axes.set_xticklabels(map(str, x_ticks))
    axes.set_yticks(y_ticks)
    axes.set_yticklabels(map(str, y_ticks))
    axes.tick_params(axis="both", which="major", pad=15)
    ################################################################################
    results_cut = _cut_result_grid(results)
    plot = axes.pcolormesh(
        coordinates_x,
        coordinates_y,
        results_cut,
        cmap=plt.get_cmap(plot_config.color_map),
        norm=normalizer,
    )
    cbar = figure.colorbar(mappable=plot, ax=axes, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(map(str, cbar_ticks))
    cbar.ax.minorticks_off()
    figure.axes[1].tick_params(labelsize=plot_config.font_size)
    return figure


def _cut_result_grid(result_grid: NPArray) -> NPArray:
    return result_grid[:-1, :-1]
