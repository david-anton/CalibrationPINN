from pathlib import Path
from typing import Any, Callable, TypeAlias, Union

import dolfinx
import gmsh
import numpy as np
import numpy.typing as npt
import petsc4py
import ufl
from dolfinx.fem import dirichletbc, locate_dofs_topological
from ufl import dot

from parametricpinn.io import ProjectDirectory

GMesh: TypeAlias = gmsh.model
GOutDimAndTags: TypeAlias = list[tuple[int, int]]
GOutDimAndTagsMap: TypeAlias = list[Union[GOutDimAndTags, list[Any]]]
GGeometry: TypeAlias = tuple[GOutDimAndTags, GOutDimAndTagsMap]
DMesh: TypeAlias = dolfinx.mesh.Mesh
DFunction: TypeAlias = dolfinx.fem.Function
DFunctionSpace: TypeAlias = dolfinx.fem.FunctionSpace
DTestFunction: TypeAlias = ufl.TestFunction
DConstant: TypeAlias = dolfinx.fem.Constant
DDofs: TypeAlias = npt.NDArray[np.int32]
DMeshTags: TypeAlias = Any  # dolfinx.mesh.MeshTags
DDirichletBC: TypeAlias = dolfinx.fem.DirichletBCMetaClass
UFLOperator: TypeAlias = ufl.core.operator.Operator
UFLMeasure: TypeAlias = ufl.Measure
UFLSigmaFunc: TypeAlias = Callable[[ufl.TrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[ufl.TrialFunction], UFLOperator]
PETScScalarType: TypeAlias = petsc4py.PETSc.ScalarType

BCValue: TypeAlias = Union[DConstant, PETScScalarType]


class NeumannBC:
    def __init__(
        self, tag: int, value: BCValue, measure: UFLMeasure, test_func: DTestFunction
    ) -> None:
        self.bc = dot(value, test_func) * measure(tag)


class DirichletBC:
    def __init__(
        self,
        tag: int,
        value: BCValue,
        dim: int,
        func_space: DFunctionSpace,
        boundary_tags: DMeshTags,
        bc_facets_dim: int,
    ) -> None:
        facet = boundary_tags.find(tag)
        dofs = locate_dofs_topological(func_space.sub(dim), bc_facets_dim, facet)
        self.bc = dirichletbc(value, dofs, func_space.sub(dim))


BoundaryConditions: TypeAlias = list[Union[DirichletBC, NeumannBC]]


def _join_output_path(
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