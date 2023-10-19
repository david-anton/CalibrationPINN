from typing import Callable, Union

import numpy as np
import ufl
from dolfinx.fem import Expression, Function, FunctionSpace, assemble_scalar, form
from mpi4py import MPI

from parametricpinn.fem.base import DFunction, UFLExpression
from parametricpinn.types import NPArray


def calculate_l2_error(
    u_approx: DFunction,
    u_exact: Union[DFunction, UFLExpression, Callable[[NPArray], NPArray]],
    degree_raise=3,
):
    # Create higher order function space (for reliable error norm computation)
    degree = u_approx.function_space.ufl_element().degree()
    family = u_approx.function_space.ufl_element().family()
    mesh = u_approx.function_space.mesh
    mpi_comm = mesh.comm
    func_space_raised = FunctionSpace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_approx_raised = Function(func_space_raised)
    u_approx_raised.interpolate(u_approx)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_exact_raised = Function(func_space_raised)
    if isinstance(u_exact, ufl.core.expr.Expr):
        interpolation_points = func_space_raised.element.interpolation_points()
        u_expression = Expression(u_exact, interpolation_points)
        u_exact_raised.interpolate(u_expression)
    else:
        u_exact_raised.interpolate(u_exact)

    # Compute the error in the higher order function space
    error_raised = Function(func_space_raised)
    error_raised.x.array[:] = u_approx_raised.x.array - u_exact_raised.x.array

    # Integrate the error
    error = form(ufl.inner(error_raised, error_raised) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mpi_comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)
