from typing import Callable, Union

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI

from parametricpinn.fem.base import DFunction
from parametricpinn.types import NPArray


def calculate_l2_error(
    u_approx: DFunction,
    u_exact: Union[DFunction, Callable[[NPArray], NPArray]],
    degree_raise: int = 3,
) -> NPArray:
    # Create higher order function space
    degree = u_approx.function_space.ufl_element().degree()
    family = u_approx.function_space.ufl_element().family()
    mesh = u_approx.function_space.mesh
    func_space_raised = fem.FunctionSpace(mesh, (family, degree + degree_raise))

    # Interpolate approximate solution
    u_approx_raised = fem.Function(func_space_raised)
    u_approx_raised.interpolate(u_approx)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_exact_raised = fem.Function(func_space_raised)
    if isinstance(u_exact, ufl.core.expr.Expr):
        u_expr = fem.Expression(u_exact, func_space_raised.element.interpolation_points)
        u_exact_raised.interpolate(u_expr)
    else:
        u_exact_raised.interpolate(u_exact)

    # Compute the error in the higher order function space
    error_raised = fem.Function(func_space_raised)
    error_raised.x.array[:] = u_approx_raised.x.array - u_exact_raised.x.array

    # Integrate the error
    error = fem.form(ufl.inner(error_raised, error_raised) * ufl.dx)
    error_local = fem.assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)
