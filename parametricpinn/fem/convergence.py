from typing import Callable, TypeAlias, Union

import numpy as np
import ufl
from dolfinx.fem import Expression, Function, FunctionSpace, assemble_scalar, form
from mpi4py import MPI

from parametricpinn.fem.base import (
    DForm,
    DFunction,
    DFunctionSpace,
    MPICommunicator,
    UFLExpression,
)
from parametricpinn.types import NPArray

UExact: TypeAlias = Union[DFunction, UFLExpression, Callable[[NPArray], NPArray]]


def calculate_l2_error(
    u_approx: DFunction,
    u_exact: UExact,
    degree_raise=3,
) -> NPArray:
    func_space_raised = _create_higher_order_function_space(u_approx, degree_raise)
    mpi_comm = func_space_raised.mesh.comm
    u_approx_raised = _interpolate_u_approx(u_approx, func_space_raised)
    u_exact_raised = _interpolate_u_exact(u_exact, func_space_raised)

    # Compute the error in the higher order function space
    error_func = _compute_error(u_approx_raised, u_exact_raised, func_space_raised)

    # Integrate the error
    error_norm = _l2_error_norm(error_func)
    return _integrate_error(error_norm, mpi_comm)


def calculate_relative_l2_error(
    u_approx: DFunction,
    u_exact: UExact,
    degree_raise=3,
) -> NPArray:
    func_space_raised = _create_higher_order_function_space(u_approx, degree_raise)
    mpi_comm = func_space_raised.mesh.comm
    u_approx_raised = _interpolate_u_approx(u_approx, func_space_raised)
    u_exact_raised = _interpolate_u_exact(u_exact, func_space_raised)

    # Compute the error in the higher order function space
    error_func = _compute_error(u_approx_raised, u_exact_raised, func_space_raised)

    # Integrate the error
    error_norm = _relative_l2_error_norm(error_func, u_exact_raised)
    return _integrate_error(error_norm, mpi_comm)


def _create_higher_order_function_space(
    u_approx: DFunction, degree_raise: int
) -> DFunctionSpace:
    degree = u_approx.function_space.ufl_element().degree()
    family = u_approx.function_space.ufl_element().family()
    mesh = u_approx.function_space.mesh
    return FunctionSpace(mesh, (family, degree + degree_raise))


def _interpolate_u_approx(u_approx: DFunction, func_space: DFunctionSpace) -> DFunction:
    func = Function(func_space)
    func.interpolate(u_approx)
    return func


def _interpolate_u_exact(u_exact: UExact, func_space: DFunctionSpace) -> DFunctionSpace:
    func = Function(func_space)
    if isinstance(u_exact, ufl.core.expr.Expr):
        interpolation_points = func_space.element.interpolation_points()
        u_expression = Expression(u_exact, interpolation_points)
        func.interpolate(u_expression)
    else:
        func.interpolate(u_exact)
    return func


def _compute_error(
    u_approx: DFunction, u_exact: DFunction, func_space: DFunctionSpace
) -> DFunction:
    func = Function(func_space)
    func.x.array[:] = u_approx.x.array - u_exact.x.array
    return func


def _integrate_error(error_norm: DForm, mpi_comm: MPICommunicator):
    error_local = assemble_scalar(error_norm)
    error_global = mpi_comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def _l2_error_norm(error_func: DFunction) -> DForm:
    return form(ufl.inner(error_func, error_func) * ufl.dx)


def _relative_l2_error_norm(error_func: DFunction, u_exact: DFunction) -> DForm:
    return form(
        (ufl.inner(error_func, error_func) / ufl.inner(u_exact, u_exact)) * ufl.dx
    )
