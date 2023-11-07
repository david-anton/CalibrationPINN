import numpy as np
import ufl
from dolfinx.fem import (
    Expression,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner


def u_ex(mod):
    return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])


u_numpy = u_ex(np)
u_ufl = u_ex(ufl)


def solve_poisson(N=10, degree=1):
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
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
    return default_problem.solve(), u_ufl(x)


def error_L2(uh, u_ex, degree_raise=3):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = FunctionSpace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def save_approximation(approximation, num_elements):
    mesh = approximation.function_space.mesh
    output_path_approx = f"output/dolfinx_report/approximation_{num_elements}.xdmf"
    output_path_solution = f"output/dolfinx_report/solution_{num_elements}.xdmf"
    func_space = approximation.function_space
    solution = Function(func_space)
    solution.interpolate(u_numpy)
    with XDMFFile(mesh.comm, output_path_approx, "w") as xdmf:
        xdmf.write_mesh(mesh)
        approximation.name = "approximation"
        xdmf.write_function(approximation)
    with XDMFFile(mesh.comm, output_path_solution, "w") as xdmf:
        xdmf.write_mesh(mesh)
        solution.name = "solution"
        xdmf.write_function(solution)


num_elements_list = [4, 8, 16]
num_elements_fine = 128
u_fine, _ = solve_poisson(N=num_elements_fine)
save_approximation(u_fine, num_elements_fine)

print("Compare to exact solution")
for num_elements in num_elements_list:
    u_approx, _ = solve_poisson(N=num_elements)
    print(f"{num_elements}: {error_L2(u_approx, u_numpy)}")

print("Compare to finely resolved approximation")
for num_elements in num_elements_list:
    u_approx, _ = solve_poisson(N=num_elements)
    save_approximation(u_approx, num_elements)
    print(f"{num_elements}: {error_L2(u_approx, u_fine)}")

print("L2-error of finely resolved approximation")
print(error_L2(u_fine, u_numpy))


# Solution
# from dolfinx.fem import (
#     Expression,
#     Function,
#     FunctionSpace,
#     assemble_scalar,
#     dirichletbc,
#     form,
#     locate_dofs_topological,
# )
# from dolfinx.io import VTXWriter
# from dolfinx.fem.petsc import LinearProblem
# from dolfinx.mesh import create_unit_square, locate_entities_boundary

# from mpi4py import MPI
# from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

# import ufl
# import numpy as np


# def u_ex(mod):
#     return lambda x: mod.cos(2 * mod.pi * x[0]) * mod.cos(2 * mod.pi * x[1])


# u_numpy = u_ex(np)
# u_ufl = u_ex(ufl)


# def solve_poisson(N=10, degree=1):
#     mesh = create_unit_square(MPI.COMM_WORLD, N, N)
#     x = SpatialCoordinate(mesh)
#     f = -div(grad(u_ufl(x)))
#     V = FunctionSpace(mesh, ("Lagrange", degree))
#     u = TrialFunction(V)
#     v = TestFunction(V)
#     a = inner(grad(u), grad(v)) * dx
#     L = f * v * dx
#     u_bc = Function(V)
#     u_bc.interpolate(u_numpy)
#     facets = locate_entities_boundary(
#         mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True)
#     )
#     dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
#     bcs = [dirichletbc(u_bc, dofs)]
#     default_problem = LinearProblem(
#         a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
#     )
#     return default_problem.solve(), u_ufl(x)


# def error_L2(uh, u_ex, degree_raise=3):
#     # Create higher order function space
#     degree = uh.function_space.ufl_element().degree()
#     family = uh.function_space.ufl_element().family()
#     mesh = uh.function_space.mesh
#     W = FunctionSpace(mesh, (family, degree + degree_raise))
#     # Interpolate approximate solution
#     u_W = Function(W)
#     u_W.interpolate(uh)

#     # Interpolate exact solution, special handling if exact solution
#     # is a ufl expression or a python lambda function
#     u_ex_W = Function(W)

#     if isinstance(u_ex, Function):
#         if u_ex.function_space.mesh != mesh:
#             from dolfinx.fem import create_nonmatching_meshes_interpolation_data
#             interpolation_data = create_nonmatching_meshes_interpolation_data(
#                 W.mesh._cpp_object,
#                 W.element,
#                 u_ex.function_space.mesh._cpp_object)
#             u_ex_W.interpolate(u_ex, nmm_interpolation_data=interpolation_data)
#         else:
#             u_ex_W.interpolate(u_ex)
#     elif isinstance(u_ex, ufl.core.expr.Expr):
#         u_expr = Expression(u_ex, W.element.interpolation_points())
#         u_ex_W.interpolate(u_expr)
#     else:
#         u_ex_W.interpolate(u_ex)

#     u_ex_W.name = "u_ex_W"
#     u_W.name = "u_W"
#     with VTXWriter(mesh.comm, f"test_u_{family}_{degree}.bp", [u_ex_W, u_W], engine="BP4") as vtx:
#         vtx.write(0.0)

#     # Compute the error in the higher order function space
#     e_W = Function(W)
#     e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

#     # Integrate the error
#     error = form(ufl.inner(e_W, e_W) * ufl.dx)
#     error_local = assemble_scalar(error)
#     error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
#     return np.sqrt(error_global)


# num_elements_list = [8, 16, 32]
# u_fine, _ = solve_poisson(N=128)

# print("Compare to exact solution")
# for num_elements in num_elements_list:
#     u_approx, _ = solve_poisson(N=num_elements)
#     print(f"{num_elements}: {error_L2(u_approx, u_numpy)}")

# print("Compare to finely resolved approximation")
# for num_elements in num_elements_list:
#     u_approx, _ = solve_poisson(N=num_elements)
#     print(f"{num_elements}: {error_L2(u_approx, u_fine)}")

# print("L2-error of finely resolved approximation")
# print(error_L2(u_fine, u_numpy))
