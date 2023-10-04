import os
from pathlib import Path
from typing import Any, TypeAlias, Union

import dolfinx
import gmsh
import numpy as np
import numpy.typing as npt
import petsc4py
import ufl

from parametricpinn.io import ProjectDirectory

GMesh: TypeAlias = gmsh.model
GOutDimAndTags: TypeAlias = list[tuple[int, int]]
GOutDimAndTagsMap: TypeAlias = list[Union[GOutDimAndTags, list[Any]]]
GGeometry: TypeAlias = tuple[GOutDimAndTags, GOutDimAndTagsMap]
DMesh: TypeAlias = dolfinx.mesh.Mesh
DFunction: TypeAlias = dolfinx.fem.Function
DFunctionSpace: TypeAlias = dolfinx.fem.FunctionSpace
DConstant: TypeAlias = dolfinx.fem.Constant
DDofs: TypeAlias = npt.NDArray[np.int32]
DMeshTags: TypeAlias = Any  # dolfinx.mesh.MeshTags
DDirichletBC: TypeAlias = dolfinx.fem.DirichletBCMetaClass
UFLOperator: TypeAlias = ufl.core.operator.Operator
UFLTrialFunction: TypeAlias = ufl.TrialFunction
UFLTestFunction: TypeAlias = ufl.TestFunction
UFLMeasure: TypeAlias = ufl.Measure
PETScScalarType: TypeAlias = petsc4py.PETSc.ScalarType
PETSLinearProblem: TypeAlias = dolfinx.fem.petsc.LinearProblem
PETSNonlinearProblem: TypeAlias = dolfinx.fem.petsc.NonlinearProblem
PETSProblem: TypeAlias = Union[PETSLinearProblem, PETSNonlinearProblem]


def join_output_path(
    project_directory: ProjectDirectory,
    file_name: str,
    output_subdir: str,
    save_to_input_dir: bool,
) -> Path:
    if save_to_input_dir:
        return project_directory.create_input_file_path(
            file_name=file_name, subdir_name=output_subdir
        )
    return project_directory.create_output_file_path(
        file_name=file_name, subdir_name=output_subdir
    )


def join_simulation_output_subdir(simulation_name: str, output_subdir: str) -> str:
    return os.path.join(output_subdir, simulation_name)
