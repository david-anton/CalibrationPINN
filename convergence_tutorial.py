from typing import Callable, TypeAlias, Union

import numpy as np
import ufl
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

from parametricpinn.fem.base import DFunction, UFLExpression
from parametricpinn.fem.convergence import (
    calculate_infinity_error,
    calculate_l2_error,
    calculate_relative_l2_error,
)
from parametricpinn.types import NPArray

UExact: TypeAlias = Union[DFunction, UFLExpression, Callable[[NPArray], NPArray]]


element_family = "Lagrange"
element_degree = 1
num_elements_reference = 128
num_elements_tests = [4, 8, 16, 32, 64]
degree_raise = 3


def u_ex(mod):
    return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])


u_numpy = u_ex(np)
u_ufl = u_ex(ufl)

communicator = MPI.COMM_WORLD


def solve_poisson(num_elements=10, degree=1):
    mesh = create_unit_square(communicator, num_elements, num_elements)
    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ufl(x)))
    V = functionspace(mesh, ("Lagrange", degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx
    u_bc = Function(V)
    u_bc.interpolate(u_numpy)
    facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True)
    )
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [dirichletbc(u_bc, dofs)]
    default_problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    return default_problem.solve()


u_exact = solve_poisson(num_elements=num_elements_reference, degree=1)


for i in range(0, len(num_elements_tests)):
    u_approx = solve_poisson(num_elements=num_elements_tests[i], degree=1)
    l2_error = calculate_l2_error(u_approx, u_exact)
    relative_l2_error = calculate_relative_l2_error(u_approx, u_exact)
    infinity_error = calculate_infinity_error(u_approx, u_exact)
    num_elements = num_elements_tests[i]
    print(
        f"{num_elements}: \t L2 error: {l2_error:.4} \t rel. L2 error: {relative_l2_error:.4} \t infinity error: {infinity_error:.4}"
    )


# u_exact = solve_poisson(num_elements_tests[-1], degree=1)

# u_approx_list = []
# for num_elements in num_elements_tests:
#     u_approx = solve_poisson(num_elements=num_elements, degree=1)
#     u_approx_list.append(u_approx)

# for i in range(len(u_approx_list)):
#     l2_error = calculate_l2_error(u_approx_list[i], u_exact, degree_raise=0)
#     # relative_l2_error = calculate_relative_l2_error(u_approx, u_refined)
#     relative_l2_error = 0.0
#     num_elements = num_elements_tests[i]
#     num_elements_refined = num_elements_tests[-1]
#     print(
#         f"{num_elements} \t -> {num_elements_refined}: \t L2 error: {l2_error:.4} \t relative L2 error: {relative_l2_error:.4}"
#     )
