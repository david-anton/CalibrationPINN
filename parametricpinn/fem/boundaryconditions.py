from typing import TypeAlias, Union

from dolfinx.fem import dirichletbc, locate_dofs_topological
from ufl import dot

from parametricpinn.fem.base import (
    DConstant,
    DFunctionSpace,
    DMeshTags,
    DTestFunction,
    PETScScalarType,
    UFLMeasure,
)

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