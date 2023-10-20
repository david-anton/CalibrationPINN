import numpy as np
import ufl
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_topological
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

from parametricpinn.fem.convergence import (
    calculate_l2_error,
    calculate_relative_l2_error,
)

element_family = "Lagrange"
element_degree = 1
num_elements_tests = [2, 4, 8, 32, 64, 128, 256, 512, 1024, 2048]
degree_raise = 3


def u_ex(mod):
    return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])


u_numpy = u_ex(np)
u_ufl = u_ex(ufl)


def solve_poisson(num_elements=10, degree=1):
    mesh = create_unit_square(MPI.COMM_WORLD, num_elements, num_elements)
    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ufl(x)))
    V = FunctionSpace(mesh, ("Lagrange", degree))
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


u_approx = solve_poisson(num_elements=num_elements_tests[0], degree=1)

for i in range(1, len(num_elements_tests)):
    u_refined = solve_poisson(num_elements=num_elements_tests[i], degree=1)
    l2_error = calculate_l2_error(u_approx, u_refined)
    relative_l2_error = calculate_relative_l2_error(u_approx, u_refined)
    num_elements = num_elements_tests[i - 1]
    num_elements_refined = num_elements_tests[i]
    print(
        f"{num_elements} \t -> {num_elements_refined}: \t rel L2 error ratio: {relative_l2_error:.4}"
    )
    u_approx = u_refined
