import os
from pathlib import Path
from typing import Any, TypeAlias, Union

import dolfinx
import gmsh
import mpi4py
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem

from calibrationpinn.io import ProjectDirectory

GMesh: TypeAlias = gmsh.model
GOutDimAndTags: TypeAlias = list[tuple[int, int]]
GOutDimAndTagsMap: TypeAlias = list[Union[GOutDimAndTags, list[Any]]]
GGeometry: TypeAlias = tuple[GOutDimAndTags, GOutDimAndTagsMap]
DMesh: TypeAlias = dolfinx.mesh.Mesh
DFunction: TypeAlias = dolfinx.fem.Function
DFunctionSpace: TypeAlias = (
    ufl.FunctionSpace
)  # dolfinx.fem.FunctionSpace not valid as a type
DScalarType: TypeAlias = dolfinx.default_scalar_type
DConstant: TypeAlias = dolfinx.fem.Constant
DDofs: TypeAlias = npt.NDArray[np.int32]
DMeshTags: TypeAlias = Any  # dolfinx.mesh.MeshTags
DDirichletBC: TypeAlias = dolfinx.fem.DirichletBC
DForm: TypeAlias = dolfinx.fem.forms.Form
UFLVariable: TypeAlias = ufl.variable
UFLOperator: TypeAlias = ufl.core.operator.Operator
UFLExpression: TypeAlias = ufl.core.expr.Expr
UFLTrialFunction: TypeAlias = ufl.TrialFunction
UFLTestFunction: TypeAlias = ufl.TestFunction
UFLMeasure: TypeAlias = ufl.Measure
PETSLinearProblem: TypeAlias = LinearProblem
PETSNonlinearProblem: TypeAlias = NonlinearProblem
PETSProblem: TypeAlias = Union[PETSLinearProblem, PETSNonlinearProblem]
MPICommunicator: TypeAlias = mpi4py.MPI.Comm


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
