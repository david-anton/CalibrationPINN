import dolfinx
import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem import (
    Constant,
    Function,
    assemble_scalar,
    create_nonmatching_meshes_interpolation_data,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_rectangle, locate_entities_boundary, meshtags
from mpi4py import MPI
from scipy import stats

# Setup
edge_length = 100.0
E = 210000.0
nu = 0.3
# FEM
fem_element_family = "Lagrange"
fem_element_degree = 1
fem_element_shape = (2,)
fem_cell_type = dolfinx.mesh.CellType.triangle  # dolfinx.mesh.CellType.quadrilateral
fem_num_elements_reference = 256
fem_num_elements_analysis = [8, 16, 32]


def solve_problem(num_elements):
    # Mesh
    mesh = create_rectangle(
        comm=MPI.COMM_WORLD,
        points=[[0.0, 0.0], [edge_length, edge_length]],
        n=[num_elements, num_elements],
        cell_type=fem_cell_type,
    )

    # Function space
    element = (fem_element_family, fem_element_degree, fem_element_shape)
    func_space = functionspace(mesh, element)
    trial_function = ufl.TrialFunction(func_space)
    test_function = ufl.TestFunction(func_space)

    # Marking facets
    tag_left = 0
    tag_right = 1
    facets_dim = mesh.topology.dim - 1
    locate_left_facet = lambda x: np.isclose(x[0], 0.0)
    locate_right_facet = lambda x: np.isclose(x[0], edge_length)
    left_facets = locate_entities_boundary(mesh, facets_dim, locate_left_facet)
    right_facets = locate_entities_boundary(mesh, facets_dim, locate_right_facet)
    marked_facets = np.hstack([left_facets, right_facets])
    marked_values = np.hstack(
        [np.full_like(left_facets, tag_left), np.full_like(right_facets, tag_right)]
    )
    sorted_facet_indices = np.argsort(marked_facets)
    facet_tags = meshtags(
        mesh,
        facets_dim,
        marked_facets[sorted_facet_indices],
        marked_values[sorted_facet_indices],
    )

    # Variational problem
    mu_ = E / (2 * (1 + nu))
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    # plane stress
    lambda_ = 2 * mu_ * lambda_ / (lambda_ + 2 * mu_)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu_ * epsilon(u) + lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(2)

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)

    volume_force = Constant(mesh, (default_scalar_type((0.0, 0.0))))
    traction_right = Constant(mesh, default_scalar_type((100.0, 0.0)))

    lhs = ufl.inner(sigma(trial_function), epsilon(test_function)) * ufl.dx
    rhs = ufl.dot(volume_force, test_function) * ufl.dx + ufl.dot(
        traction_right, test_function
    ) * ds(tag_right)

    u_left = Constant(mesh, default_scalar_type((0.0, 0.0)))
    dofs_facet_left = locate_dofs_topological(func_space, facets_dim, left_facets)
    u_bc_left = dirichletbc(u_left, dofs_facet_left, func_space)

    problem = LinearProblem(
        lhs,
        rhs,
        bcs=[u_bc_left],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )

    return problem.solve()


# def save_function(approximation, num_elements):
#     mesh = approximation.function_space.mesh
#     output_path = f"output/dolfinx_report/approximation_{num_elements}.xdmf"
#     with XDMFFile(mesh.comm, output_path, "w") as xdmf:
#         xdmf.write_mesh(mesh)
#         approximation.name = "approximation"
#         xdmf.write_function(approximation)


# import matplotlib.pyplot as plt
# import numpy as np
# from scipy import stats

# from parametricpinn.io import ProjectDirectory
# from parametricpinn.settings import Settings


# def plot_error_convergence_analysis(
#     error_record: list[float],
#     element_size_record: list[float],
#     error_norm: str,
#     output_subdirectory: str,
#     project_directory: ProjectDirectory,
# ) -> None:
#     figure, axes = plt.subplots()
#     log_element_size = np.log(np.array(element_size_record))
#     log_error = np.log(np.array(error_record))

#     slope, intercept, _, _, _ = stats.linregress(log_element_size, log_error)
#     regression_log_error = slope * log_element_size + intercept

#     axes.plot(log_element_size, log_error, "ob", label="simulation")
#     axes.plot(log_element_size, regression_log_error, "--r", label="regression")
#     axes.set_xlabel("log element size")
#     axes.set_ylabel(f"log error {error_norm}")
#     axes.set_title("Convergence analysis")
#     axes.legend(loc="best")
#     axes.text(
#         log_element_size[1],
#         log_error[-1],
#         f"convergence rate: {slope:.6}",
#         style="italic",
#         bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
#     )
#     file_name = f"convergence_log_{error_norm}_error.png"
#     output_path = project_directory.create_output_file_path(
#         file_name, output_subdirectory
#     )
#     figure.savefig(output_path, bbox_inches="tight", dpi=256)
#     plt.clf()


def L2_error(u_approx, u_exact, degree_raise=3):
    # Create higher order function space
    degree = u_approx.function_space.ufl_element().degree()
    family = u_approx.function_space.ufl_element().family()
    mesh = u_approx.function_space.mesh
    element = (family, degree + degree_raise, fem_element_shape)
    W = functionspace(mesh, element)

    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(u_approx)

    # Interpolate exact solution
    u_ex_W = Function(W)
    interpolation_data = create_nonmatching_meshes_interpolation_data(
        W.mesh._cpp_object, W.element, u_exact.function_space.mesh._cpp_object
    )
    u_ex_W.interpolate(u_exact, nmm_interpolation_data=interpolation_data)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def calculate_convegrence_order(errors, element_sizes):
    log_element_sizes = np.log(np.array(element_sizes))
    log_errors = np.log(np.array(errors))
    slope, _, _, _, _ = stats.linregress(log_element_sizes, log_errors)
    return slope


u_exact = solve_problem(fem_num_elements_reference)

element_sizes = []
l2_errors = []

for num_elements in fem_num_elements_analysis:
    u_approx = solve_problem(num_elements)
    element_sizes.append(edge_length / num_elements)
    # save_function(u_approx, num_elements)
    l2_errors.append(L2_error(u_approx, u_exact))

convergence_order = calculate_convegrence_order(l2_errors, element_sizes)
# plot_error_convergence_analysis(
#     l2_errors,
#     element_sizes,
#     error_norm="l2",
#     output_subdirectory="output",
#     project_directory=ProjectDirectory(Settings()),
# )
print(f"Convergence order: {convergence_order}")
