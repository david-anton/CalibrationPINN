import dolfinx.fem as fem
import ufl

from parametricpinn.fem.base import DFunction, DFunctionSpace, DMesh, UFLVariable


def compute_infinitesimal_strain_function(
    solution_function: DFunction, mesh: DMesh
) -> DFunction:
    H = ufl.variable(ufl.grad(solution_function))
    H_transposed = ufl.variable(ufl.transpose(H))
    epsilon = ufl.variable(1 / 2 * (H + H_transposed))
    return project_strain_function(epsilon, solution_function, mesh)


def compute_green_strain_function(
    solution_function: DFunction, mesh: DMesh
) -> DFunction:
    dim = solution_function.function_space.num_sub_spaces
    I = ufl.variable(ufl.Identity(dim))
    F = ufl.variable(ufl.grad(solution_function) + I)
    F_transposed = ufl.variable(ufl.transpose(F))
    C = ufl.variable(F_transposed * F)
    E = ufl.variable(1 / 2 * (C - I))
    return project_strain_function(E, solution_function, mesh)


def project_strain_function(
    strain: UFLVariable, solution_function: DFunction, mesh: DMesh
) -> DFunction:
    strain_func_space = create_strain_function_space(solution_function, mesh)
    strain_expression = fem.Expression(
        strain, strain_func_space.element.interpolation_points()
    )
    strain_function = fem.Function(strain_func_space)
    strain_function.interpolate(strain_expression)
    return strain_function


def create_strain_function_space(
    solution_function: DFunction, mesh: DMesh
) -> DFunctionSpace:
    degree_solution = solution_function.function_space.ufl_element().degree()
    family_solution = solution_function.function_space.ufl_element().family()
    num_sub_spaces = solution_function.function_space.num_sub_spaces
    shape = (num_sub_spaces, num_sub_spaces) if num_sub_spaces != 0 else None
    element = (family_solution, degree_solution - 1, shape)
    return fem.functionspace(mesh, element)
