from typing import TypeAlias, Union

from dolfinx.fem import dirichletbc, locate_dofs_topological
from ufl import dot

from parametricpinn.fem.base import (
    DConstant,
    DFunctionSpace,
    DMeshTags,
    PETScScalarType,
    UFLMeasure,
    UFLTestFunction,
)

BCValue: TypeAlias = Union[DConstant, PETScScalarType]


class NeumannBC:
    def __init__(
        self,
        tag: int,
        value: BCValue,
        measure: UFLMeasure,
        test_function: UFLTestFunction,
    ) -> None:
        self.bc = dot(value, test_function) * measure(tag)


class DirichletBC:
    def __init__(
        self,
        tag: int,
        value: BCValue,
        dim: int,
        function_space: DFunctionSpace,
        boundary_tags: DMeshTags,
        bc_facets_dim: int,
    ) -> None:
        facet = boundary_tags.find(tag)
        dofs = locate_dofs_topological(function_space.sub(dim), bc_facets_dim, facet)
        self.bc = dirichletbc(value, dofs, function_space.sub(dim))


BoundaryConditions: TypeAlias = list[Union[DirichletBC, NeumannBC]]
