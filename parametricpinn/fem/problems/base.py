from parametricpinn.fem.base import DDirichletBC, UFLOperator
from parametricpinn.fem.boundaryconditions import BoundaryConditions, DirichletBC


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
