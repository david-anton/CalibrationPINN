from dataclasses import dataclass

import numpy as np
import pandas as pd
from dolfinx.io import XDMFFile

from parametricpinn.fem.base import (
    DDirichletBC,
    DFunction,
    DMesh,
    UFLOperator,
    join_output_path,
)
from parametricpinn.fem.boundaryconditions import BoundaryConditions, DirichletBC
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.types import NPArray


@dataclass
class BaseSimulationResults:
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
