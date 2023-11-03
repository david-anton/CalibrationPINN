from typing import Callable, TypeAlias, Union

import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Expression,
    Function,
    assemble_scalar,
    create_nonmatching_meshes_interpolation_data,
    form,
    functionspace,
)
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


def l2_error(
    u_approx: DFunction,
    u_exact: UExact,
    degree_raise=3,
) -> float:
    func_space = _create_higher_order_function_space(u_approx, degree_raise)
    mpi_comm = _get_mpi_communicator(func_space)
    u_approx_interpolated = _interpolate_u_approx(u_approx, func_space)
    u_exact_interpolated = _interpolate_u_exact(u_exact, func_space)

    error_func = _compute_error(u_approx_interpolated, u_exact_interpolated, func_space)

    error_norm = _l2_error_norm(error_func)
    return _integrate_error(error_norm, mpi_comm).item()


def relative_l2_error(
    u_approx: DFunction,
    u_exact: UExact,
    degree_raise=3,
) -> float:
    func_space = _create_higher_order_function_space(u_approx, degree_raise)
    mpi_comm = _get_mpi_communicator(func_space)
    u_approx_interpolated = _interpolate_u_approx(u_approx, func_space)
    u_exact_interpolated = _interpolate_u_exact(u_exact, func_space)

    error_func = _compute_error(u_approx_interpolated, u_exact_interpolated, func_space)

    error_norm = _relative_l2_error_norm(error_func, u_exact_interpolated)
    return _integrate_error(error_norm, mpi_comm).item()


def h01_error(
    u_approx: DFunction,
    u_exact: UExact,
    degree_raise=3,
) -> float:
    func_space = _create_higher_order_function_space(u_approx, degree_raise)
    mpi_comm = _get_mpi_communicator(func_space)
    u_approx_interpolated = _interpolate_u_approx(u_approx, func_space)
    u_exact_interpolated = _interpolate_u_exact(u_exact, func_space)

    error_func = _compute_error(u_approx_interpolated, u_exact_interpolated, func_space)

    error_norm = _h01_error_norm(error_func)
    return _integrate_error(error_norm, mpi_comm).item()


def infinity_error(
    u_approx: DFunction,
    u_exact: UExact,
) -> float:
    func_space = u_approx.function_space
    mpi_comm = _get_mpi_communicator(func_space)
    u_exact_interpolated = _interpolate_u_exact(u_exact, func_space)

    error_norm = _infinity_error_norm(u_approx, u_exact_interpolated, mpi_comm)
    return error_norm.item()


def _create_higher_order_function_space(
    u_approx: DFunction, degree_raise: int
) -> DFunctionSpace:
    degree = u_approx.function_space.ufl_element().degree()
    family = u_approx.function_space.ufl_element().family()
    mesh = u_approx.function_space.mesh
    num_sub_spaces = u_approx.function_space.num_sub_spaces
    shape = (num_sub_spaces,) if num_sub_spaces != 0 else None
    element = (family, degree + degree_raise, shape)
    return functionspace(mesh, element)


def _interpolate_u_approx(u_approx: DFunction, func_space: DFunctionSpace) -> DFunction:
    func = Function(func_space, dtype=default_scalar_type)
    func.interpolate(u_approx)
    return func


def _get_mpi_communicator(func_space: DFunctionSpace) -> MPICommunicator:
    return func_space.mesh.comm


def _interpolate_u_exact(u_exact: UExact, func_space: DFunctionSpace) -> DFunctionSpace:
    func = Function(func_space, dtype=default_scalar_type)
    if isinstance(u_exact, DFunction):
        func.interpolate(
            u_exact,
            nmm_interpolation_data=create_nonmatching_meshes_interpolation_data(
                func.function_space.mesh._cpp_object,
                func.function_space.element,
                u_exact.function_space.mesh._cpp_object,
            ),
        )
    elif isinstance(u_exact, ufl.core.expr.Expr):
        interpolation_points = func_space.element.interpolation_points()
        u_expression = Expression(u_exact, interpolation_points)
        func.interpolate(u_expression)
    else:
        func.interpolate(u_exact)
    return func


def _compute_error(
    u_approx: DFunction, u_exact: DFunction, func_space: DFunctionSpace
) -> DFunction:
    func = Function(func_space, dtype=default_scalar_type)
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


def _h01_error_norm(error_func: DFunction) -> DForm:
    return form(ufl.dot(ufl.grad(error_func), ufl.grad(error_func)) * ufl.dx)


def _infinity_error_norm(
    u_approx: DFunction, u_exact: DFunction, mpi_comm: MPICommunicator
) -> NPArray:
    error_local = np.max(np.abs(u_approx.x.array - u_exact.x.array))
    error_global = mpi_comm.allreduce(error_local, op=MPI.MAX)
    return error_global
