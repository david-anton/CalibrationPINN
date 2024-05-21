from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd
from dolfinx.io import XDMFFile

from calibrationpinn.errors import FEMResultsError
from calibrationpinn.fem.base import (
    DDirichletBC,
    DFunction,
    DMesh,
    UFLOperator,
    join_output_path,
)
from calibrationpinn.fem.boundaryconditions import BoundaryConditions, DirichletBC
from calibrationpinn.fem.mechanics import (
    compute_green_strain_function,
    compute_infinitesimal_strain_function,
)
from calibrationpinn.io import ProjectDirectory
from calibrationpinn.io.readerswriters import PandasDataWriter
from calibrationpinn.types import NPArray

MaterialParameterNames: TypeAlias = tuple[str, ...]
MaterialParameters: TypeAlias = tuple[float, ...]


@dataclass
class BaseSimulationResults:
    material_parameter_names: MaterialParameterNames
    material_parameters: MaterialParameters
    coordinates_x: NPArray
    coordinates_y: NPArray
    displacements_x: NPArray
    displacements_y: NPArray
    function: DFunction


def apply_boundary_conditions(
    boundary_conditions: BoundaryConditions, rhs: UFLOperator
) -> tuple[list[DDirichletBC], UFLOperator]:
    dirichlet_bcs = []
    for condition in boundary_conditions:
        if isinstance(condition, DirichletBC):
            dirichlet_bcs.append(condition.bc)
        else:
            rhs += condition.bc
    return dirichlet_bcs, rhs


def save_displacements(
    results: BaseSimulationResults,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    data_writer = PandasDataWriter(project_directory)
    file_name = "displacements"
    results_dict = {
        "coordinates_x": np.ravel(results.coordinates_x),
        "coordinates_y": np.ravel(results.coordinates_y),
        "displacements_x": np.ravel(results.displacements_x),
        "displacements_y": np.ravel(results.displacements_y),
    }
    results_dataframe = pd.DataFrame(results_dict)
    data_writer.write(
        results_dataframe,
        file_name,
        output_subdir,
        header=True,
        save_to_input_dir=save_to_input_dir,
    )


def save_parameters(
    simulation_results: BaseSimulationResults,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    _validate_results(simulation_results)
    data_writer = PandasDataWriter(project_directory)
    file_name = "parameters"
    results = simulation_results
    num_parameters = len(results.material_parameter_names)
    results_dict = {
        results.material_parameter_names[i]: np.array([results.material_parameters[i]])
        for i in range(num_parameters)
    }
    results_dataframe = pd.DataFrame(results_dict)
    data_writer.write(
        results_dataframe,
        file_name,
        output_subdir,
        header=True,
        save_to_input_dir=save_to_input_dir,
    )


def save_results_as_xdmf(
    mesh: DMesh,
    approximate_solution: DFunction,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> None:
    file_name = "displacements.xdmf"
    output_path = join_output_path(
        project_directory, file_name, output_subdir, save_to_input_dir
    )
    with XDMFFile(mesh.comm, output_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        # approximate_solution.name = "approximation"
        # xdmf.write_function(approximate_solution)


def _validate_results(simulation_results: BaseSimulationResults) -> None:
    results = simulation_results
    if not len(results.material_parameter_names) == len(results.material_parameters):
        raise FEMResultsError(
            "The number of parameter names does not correspond to the number of parameters."
        )


def print_maximum_infinitesiaml_strain(
    solution_function: DFunction, mesh: DMesh
) -> None:
    strain_function = compute_infinitesimal_strain_function(solution_function, mesh)
    geometric_dim = mesh.geometry.dim
    strain = strain_function.x.array.reshape((-1, geometric_dim, geometric_dim))
    max_strain_xx = np.amax(np.absolute(strain[:, 0, 0]))
    max_strain_yy = np.amax(np.absolute(strain[:, 1, 1]))
    max_strain_xy = np.amax(np.absolute(strain[:, 0, 1]))
    max_strain_yx = np.amax(np.absolute(strain[:, 1, 0]))
    print(
        f"Maximum infinitesimal strains: eps_xx = {max_strain_xx}, eps_yy = {max_strain_yy}, eps_xy = {max_strain_xy}, eps_yx = {max_strain_yx}"
    )


def print_maximum_green_strain(solution_function: DFunction, mesh: DMesh) -> None:
    strain_function = compute_green_strain_function(solution_function, mesh)
    geometric_dim = mesh.geometry.dim
    strain = strain_function.x.array.reshape((-1, geometric_dim, geometric_dim))
    max_strain_xx = np.amax(np.absolute(strain[:, 0, 0]))
    max_strain_yy = np.amax(np.absolute(strain[:, 1, 1]))
    max_strain_xy = np.amax(np.absolute(strain[:, 0, 1]))
    max_strain_yx = np.amax(np.absolute(strain[:, 1, 0]))
    print(
        f"Maximum Green strains: E_xx = {max_strain_xx}, E_yy = {max_strain_yy}, E_xy = {max_strain_xy}, E_yx = {max_strain_yx}"
    )
