from collections import namedtuple
from typing import TypeAlias, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

from parametricpinn.errors import FEMDomainConfigError, PlottingConfigError
from parametricpinn.fem import (
    LinearElasticityProblemConfig,
    NeoHookeanProblemConfig,
    PlateWithHoleDomainConfig,
    ProblemConfigs,
    QuarterPlateWithHoleDomainConfig,
)
from parametricpinn.fem import SimulationConfig as FEMSimulationConfig
from parametricpinn.fem import run_simulation
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import Device, Module, NPArray, PLTAxes, PLTFigure

ProblemConfigLists: TypeAlias = Union[
    list[LinearElasticityProblemConfig], list[NeoHookeanProblemConfig]
]
DomainConfigs: TypeAlias = Union[
    QuarterPlateWithHoleDomainConfig, PlateWithHoleDomainConfig
]


class DisplacementsPlotterConfig2D:
    def __init__(self) -> None:
        # label size
        self.label_size = 16
        # font size in legend
        self.font_size = 16
        self.font = {"size": self.label_size}
        # title pad
        self.title_pad = 10
        # labels
        self.x_label = "x [mm]"
        self.y_label = "y [mm]"
        # major ticks
        self.major_tick_label_size = 16
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
        self.num_points_per_edge = 128
        # save options
        self.dpi = 300
        self.file_format = "pdf"


SimulationConfig = namedtuple(
    "SimulationConfig",
    [
        "domain_config",
        "problem_config",
        "volume_force_x",
        "volume_force_y",
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


def plot_displacements_2d(
    ansatz: Module,
    domain_config: DomainConfigs,
    problem_configs: ProblemConfigLists,
    volume_force_x: float,
    volume_force_y: float,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfig2D,
    device: Device,
) -> None:
    ansatz.eval()
    for problem_config in problem_configs:
        simulation_config = SimulationConfig(
            domain_config=domain_config,
            problem_config=problem_config,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
        )
        _plot_one_simulation(
            ansatz,
            simulation_config,
            output_subdir,
            project_directory,
            plot_config,
            device,
        )


def _plot_one_simulation(
    ansatz: Module,
    simulation_config: SimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfig2D,
    device: Device,
):
    simulation_results = _calculate_results(
        ansatz, simulation_config, output_subdir, project_directory, device
    )
    _plot_results(
        simulation_results,
        simulation_config,
        output_subdir,
        project_directory,
        plot_config,
    )


def _calculate_results(
    ansatz: Module,
    simulation_config: SimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    device: Device,
) -> SimulationResults:
    (
        coordinates_x,
        coordinates_y,
        fem_displacements_x,
        fem_displacements_y,
    ) = _run_simulation(simulation_config, output_subdir, project_directory)
    pinn_displacements_x, pinn_displacements_y = _run_prametric_pinn(
        ansatz, simulation_config, coordinates_x, coordinates_y, device
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
    fem_simulation_config = FEMSimulationConfig(
        domain_config=simulation_config.domain_config,
        problem_config=simulation_config.problem_config,
        volume_force_x=simulation_config.volume_force_x,
        volume_force_y=simulation_config.volume_force_y,
    )
    results = run_simulation(
        simulation_config=fem_simulation_config,
        save_results=False,
        save_metadata=False,
        output_subdir=output_subdir,
        project_directory=project_directory,
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
    device: Device,
) -> tuple[NPArray, NPArray]:
    parameters = _get_parameters_from_problem(simulation_config.problem_config)
    coordinates = torch.concat(
        (torch.from_numpy(coordinates_x), torch.from_numpy(coordinates_y)), dim=1
    )
    inputs = torch.concat(
        (
            coordinates,
            torch.from_numpy(parameters).repeat(coordinates.size(dim=0), 1),
        ),
        dim=1,
    ).to(device)
    with torch.no_grad():
        outputs = ansatz(inputs)
    outputs = outputs.detach().cpu().numpy()
    displacements_x = outputs[:, 0].reshape((-1, 1))
    displacements_y = outputs[:, 1].reshape((-1, 1))
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
    simulation_config: SimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfig2D,
) -> None:
    _plot_simulation_and_prediction(
        pinn_displacements=simulation_results.pinn_displacements_x,
        fem_displacements=simulation_results.fem_displacements_x,
        coordinates_x=simulation_results.coordinates_x,
        coordinates_y=simulation_results.coordinates_y,
        title=r"Displacements $u_{x}$",
        file_name_identifier="displacements_x",
        output_subdir=output_subdir,
        project_directory=project_directory,
        plot_config=plot_config,
        simulation_config=simulation_config,
    )
    _plot_simulation_and_prediction(
        pinn_displacements=simulation_results.pinn_displacements_y,
        fem_displacements=simulation_results.fem_displacements_y,
        coordinates_x=simulation_results.coordinates_x,
        coordinates_y=simulation_results.coordinates_y,
        title=r"Displacements $u_{y}$",
        file_name_identifier="displacements_y",
        output_subdir=output_subdir,
        project_directory=project_directory,
        plot_config=plot_config,
        simulation_config=simulation_config,
    )
    _plot_errors(
        errors=simulation_results.re_displacements_x,
        coordinates_x=simulation_results.coordinates_x,
        coordinates_y=simulation_results.coordinates_y,
        title=r"Relative error $u_{x}$",
        file_name_identifier="relative_error_x",
        output_subdir=output_subdir,
        project_directory=project_directory,
        plot_config=plot_config,
        simulation_config=simulation_config,
    )
    _plot_errors(
        errors=simulation_results.re_displacements_y,
        coordinates_x=simulation_results.coordinates_x,
        coordinates_y=simulation_results.coordinates_y,
        title=r"Relative error $u_{y}$",
        file_name_identifier="relative_error_y",
        output_subdir=output_subdir,
        project_directory=project_directory,
        plot_config=plot_config,
        simulation_config=simulation_config,
    )


def _plot_simulation_and_prediction(
    pinn_displacements: NPArray,
    fem_displacements: NPArray,
    coordinates_x: NPArray,
    coordinates_y: NPArray,
    title: str,
    file_name_identifier: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfig2D,
    simulation_config: SimulationConfig,
) -> None:
    composed_results = np.concatenate((pinn_displacements, fem_displacements), axis=0)
    normalizer = _create_normalizer(composed_results, plot_config)
    ticks = _create_ticks(composed_results, plot_config)
    coordinates_grid_x, coordinates_grid_y = _generate_coordinate_grid(
        simulation_config, plot_config
    )
    interpolated_pinn_displacements = _interpolate_results_on_grid(
        pinn_displacements,
        coordinates_x,
        coordinates_y,
        coordinates_grid_x,
        coordinates_grid_y,
    )
    interpolated_fem_displacements = _interpolate_results_on_grid(
        fem_displacements,
        coordinates_x,
        coordinates_y,
        coordinates_grid_x,
        coordinates_grid_y,
    )
    figure_pinn = _plot_once(
        interpolated_pinn_displacements,
        coordinates_grid_x,
        coordinates_grid_y,
        title,
        normalizer,
        ticks,
        plot_config,
        simulation_config,
    )
    figure_fem = _plot_once(
        interpolated_fem_displacements,
        coordinates_grid_x,
        coordinates_grid_y,
        title,
        normalizer,
        ticks,
        plot_config,
        simulation_config,
    )
    parameter_prefix = _get_file_name_parameter_prefix_from_problem(
        simulation_config.problem_config
    )
    file_name_pinn = (
        f"plot_{file_name_identifier}_PINN_{parameter_prefix}.{plot_config.file_format}"
    )
    file_name_fem = (
        f"plot_{file_name_identifier}_FEM_{parameter_prefix}.{plot_config.file_format}"
    )
    _save_plot(
        figure_pinn, file_name_pinn, output_subdir, project_directory, plot_config
    )
    _save_plot(figure_fem, file_name_fem, output_subdir, project_directory, plot_config)


def _plot_errors(
    errors: NPArray,
    coordinates_x: NPArray,
    coordinates_y: NPArray,
    title: str,
    file_name_identifier: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfig2D,
    simulation_config: SimulationConfig,
) -> None:
    normalizer = _create_normalizer(errors, plot_config)
    ticks = _create_ticks(errors, plot_config)
    coordinates_grid_x, coordinates_grid_y = _generate_coordinate_grid(
        simulation_config, plot_config
    )
    interpolated_errors = _interpolate_results_on_grid(
        errors,
        coordinates_x,
        coordinates_y,
        coordinates_grid_x,
        coordinates_grid_y,
    )
    figure = _plot_once(
        interpolated_errors,
        coordinates_grid_x,
        coordinates_grid_y,
        title,
        normalizer,
        ticks,
        plot_config,
        simulation_config,
    )
    parameter_prefix = _get_file_name_parameter_prefix_from_problem(
        simulation_config.problem_config
    )
    file_name = (
        f"plot_{file_name_identifier}_{parameter_prefix}.{plot_config.file_format}"
    )
    _save_plot(figure, file_name, output_subdir, project_directory, plot_config)


def _create_normalizer(
    results: NPArray, plot_config: DisplacementsPlotterConfig2D
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
    results: NPArray, plot_config: DisplacementsPlotterConfig2D
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
    simulation_config: SimulationConfig,
    plot_config: DisplacementsPlotterConfig2D,
) -> list[NPArray]:
    grid_coordinates_x = np.linspace(
        -simulation_config.domain_config.edge_length,
        0.0,
        num=plot_config.num_points_per_edge,
    )
    grid_coordinates_y = np.linspace(
        0.0,
        simulation_config.domain_config.edge_length,
        num=plot_config.num_points_per_edge,
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
        method="cubic",
    )


def _plot_once(
    results: NPArray,
    coordinates_x: NPArray,
    coordinates_y: NPArray,
    title: str,
    normalizer: BoundaryNorm,
    cbar_ticks: list[float],
    plot_config: DisplacementsPlotterConfig2D,
    simulation_config: SimulationConfig,
) -> PLTFigure:
    figure, axes = plt.subplots()

    def _set_title_and_labels(axes: PLTAxes) -> None:
        axes.set_title(title, pad=plot_config.title_pad, **plot_config.font)
        axes.set_xlabel(plot_config.x_label, **plot_config.font)
        axes.set_ylabel(plot_config.y_label, **plot_config.font)
        axes.tick_params(
            axis="both", which="minor", labelsize=plot_config.minor_tick_label_size
        )
        axes.tick_params(
            axis="both", which="major", labelsize=plot_config.major_tick_label_size
        )

    def _configure_ticks(axes: PLTAxes) -> None:
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

    def _configure_color_bar(figure: PLTFigure) -> None:
        cbar = figure.colorbar(mappable=plot, ax=axes, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(map(str, cbar_ticks))
        cbar.ax.minorticks_off()
        figure.axes[1].tick_params(labelsize=plot_config.font_size)

    # def _add_hole(axes: PLTAxes) -> None:
    #     hole = plt.Circle(
    #         (0.0, 0.0),
    #         radius=simulation_config.domain_config.radius,
    #         color="white",
    #     )
    #     axes.add_patch(hole)

    _set_title_and_labels(axes)
    _configure_ticks(axes)
    results_cut = _cut_result_grid(results)
    plot = axes.pcolormesh(
        coordinates_x,
        coordinates_y,
        results_cut,
        cmap=plt.get_cmap(plot_config.color_map),
        norm=normalizer,
    )
    _configure_color_bar(figure)
    _add_geometry_specific_patches(axes, simulation_config)
    return figure


def _add_geometry_specific_patches(
    axes: PLTAxes, simulation_config: SimulationConfig
) -> None:
    def add_quarter_hole(
        axes: PLTAxes, domain_config: QuarterPlateWithHoleDomainConfig
    ):
        radius = domain_config.radius
        hole = plt.Circle(
            (0.0, 0.0),
            radius=radius,
            color="white",
        )
        axes.add_patch(hole)

    def add_hole(axes: PLTAxes, domain_config: PlateWithHoleDomainConfig):
        length = domain_config.plate_length
        height = domain_config.plate_height
        radius = domain_config.hole_radius
        hole = plt.Circle(
            (length / 2, height / 2),
            radius=radius,
            color="white",
        )
        axes.add_patch(hole)

    domain_config = simulation_config.domain_config
    if isinstance(domain_config, QuarterPlateWithHoleDomainConfig):
        add_quarter_hole(axes=axes, domain_config=domain_config)
    elif isinstance(domain_config, PlateWithHoleDomainConfig):
        add_hole(axes=axes, domain_config=domain_config)
    else:
        raise FEMDomainConfigError(
            f"There is no implementation for the requested FEM domain {domain_config}."
        )


def _cut_result_grid(result_grid: NPArray) -> NPArray:
    return result_grid[:-1, :-1]


def _save_plot(
    figure: PLTFigure,
    file_name: str,
    output_subdir: str,
    project_directory: ProjectDirectory,
    plot_config: DisplacementsPlotterConfig2D,
) -> None:
    save_path = project_directory.create_output_file_path(file_name, output_subdir)
    figure.savefig(
        save_path,
        format=plot_config.file_format,
        bbox_inches="tight",
        dpi=plot_config.dpi,
    )


def _get_parameters_from_problem(problem_config: ProblemConfigs) -> NPArray:
    if isinstance(problem_config, LinearElasticityProblemConfig):
        return np.array([problem_config.youngs_modulus, problem_config.poissons_ratio])
    elif isinstance(problem_config, NeoHookeanProblemConfig):
        return np.array([problem_config.youngs_modulus, problem_config.poissons_ratio])
    else:
        raise PlottingConfigError(
            f"There is no implementation for the requested FEM problem config {problem_config}."
        )


def _get_file_name_parameter_prefix_from_problem(problem_config: ProblemConfigs) -> str:
    if isinstance(problem_config, LinearElasticityProblemConfig):
        youngs_modulus = round(problem_config.youngs_modulus, 2)
        poissons_ratio = round(problem_config.poissons_ratio, 4)
        return f"E_{youngs_modulus}_nu_{poissons_ratio}"
    elif isinstance(problem_config, NeoHookeanProblemConfig):
        youngs_modulus = round(problem_config.youngs_modulus, 2)
        poissons_ratio = round(problem_config.poissons_ratio, 4)
        return f"E_{youngs_modulus}_nu_{poissons_ratio}"
    else:
        raise PlottingConfigError(
            f"There is no implementation for the requested FEM problem config {problem_config}."
        )