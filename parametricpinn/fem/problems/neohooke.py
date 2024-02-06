from dataclasses import dataclass
from typing import Callable, TypeAlias

import dolfinx
import dolfinx.fem as fem
import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from parametricpinn.fem.base import (
    DConstant,
    DFunction,
    DFunctionSpace,
    PETSNonlinearProblem,
    UFLOperator,
    UFLTrialFunction,
)
from parametricpinn.fem.domains import Domain
from parametricpinn.fem.mechanics import (
    compute_green_strain_function,
    compute_infinitesimal_strain_function,
)
from parametricpinn.fem.problems.base import (
    BaseSimulationResults,
    MaterialParameters,
    apply_boundary_conditions,
    save_displacements,
    save_parameters,
)
from parametricpinn.io import ProjectDirectory

UFLSigmaFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]


@dataclass
class NeoHookeProblemConfig:
    material_parameter_names = ("bulk modulus", "shear modulus")
    material_parameters: MaterialParameters


@dataclass
class NeoHookeResults(BaseSimulationResults):
    pass


class NeoHookeProblem:
    def __init__(
        self,
        config: NeoHookeProblemConfig,
        domain: Domain,
        function_space: DFunctionSpace,
        volume_force: DConstant,
    ):
        self._config = config
        self._domain = domain
        self._mesh = domain.mesh
        self._function_space = function_space
        self._volume_force = volume_force
        self._quadrature_degree = 4
        self._problem, self._solution_function = self._define()

    def solve(self) -> DFunction:
        solver = NewtonSolver(self._mesh.comm, self._problem)

        # Set Newton solver options
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.max_it = 20
        solver.convergence_criterion = "incremental"

        solver.report = True
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        # dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

        num_iterations, converged = solver.solve(self._solution_function)
        assert converged
        print(f"Number of iterations: {num_iterations}")
        self._print_maximum_infinitesiaml_strain(self._solution_function)
        self._print_maximum_green_strain(self._solution_function)
        return self._solution_function

    def compile_results(self, approximate_solution: DFunction) -> NeoHookeResults:
        coordinates = self._function_space.tabulate_dof_coordinates()
        coordinates_x = coordinates[:, 0].reshape((-1, 1))
        coordinates_y = coordinates[:, 1].reshape((-1, 1))

        displacements = approximate_solution.x.array.reshape(
            (-1, self._mesh.geometry.dim)
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))

        results = NeoHookeResults(
            material_parameter_names=self._config.material_parameter_names,
            material_parameters=self._config.material_parameters,
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            displacements_x=displacements_x,
            displacements_y=displacements_y,
            function=approximate_solution,
        )

        return results

    def save_results(
        self,
        results: BaseSimulationResults,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ) -> None:
        save_displacements(results, output_subdir, save_to_input_dir, project_directory)
        save_parameters(results, output_subdir, save_to_input_dir, project_directory)

    def _define(self) -> tuple[PETSNonlinearProblem, DFunction]:
        ### Plane strain assumed
        # Solution and test function
        test_function = ufl.TestFunction(self._function_space)
        solution_function = fem.Function(self._function_space)

        # Deformation gradient
        I_2D = ufl.variable(ufl.Identity(2))
        F_2D = ufl.variable(ufl.grad(solution_function) + I_2D)
        F = ufl.variable(
            ufl.as_matrix(
                [
                    [F_2D[0, 0], F_2D[0, 1], 0.0],
                    [F_2D[1, 0], F_2D[1, 1], 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

        # Right Cauchy-Green tensor
        F_transpose = ufl.variable(ufl.transpose(F))
        C = ufl.variable(F_transpose * F)
        J = ufl.variable(ufl.det(C) ** (1 / 2))  # J = ufl.variable(ufl.det(F))

        # Material parameters
        K = fem.Constant(
            self._mesh, default_scalar_type(self._config.material_parameters[0])
        )
        G = fem.Constant(
            self._mesh, default_scalar_type(self._config.material_parameters[1])
        )
        c_10 = G / 2

        # Isochoric deformation tensors and invariants
        C_iso = ufl.variable((J ** (-2 / 3)) * C)  # Isochoric right Cauchy-Green tensor
        I_C_iso = ufl.variable(ufl.tr(C_iso))  # Isochoric first invariant

        ### Strain Energy
        # Volumetric part of strain energy
        # W_vol = (K / 2) * ((J - 1) ** 2)
        W_vol = (K / 4) * (J**2 - 1 - 2 * ufl.ln(J))
        # Isochoric part of strain energy
        W_iso = c_10 * (I_C_iso - 3)
        W = W_vol + W_iso

        # 2. Piola-Kirchoff stress tensor
        T = 2 * ufl.diff(W, C)
        # I_3D = ufl.variable(ufl.Identity(3))
        # C_iso_inverse = ufl.inv(C_iso)
        # T = (J ** (1 / 3)) * K * (J - 1) * C_iso_inverse + 2 * (J ** (-2 / 3)) * (
        #     c_10 * I_3D - (1 / 3) * c_10 * I_C_iso * C_iso_inverse
        # )

        # 1. Piola-Kirchoff stress tensor
        P = ufl.variable(F * T)
        P_2D = ufl.as_matrix([[P[0, 0], P[0, 1]], [P[1, 0], P[1, 1]]])

        # Define variational form
        metadata = {"quadrature_degree": self._quadrature_degree}
        boundary_tags = self._domain.boundary_tags
        ds = ufl.Measure(
            "ds", domain=self._mesh, subdomain_data=boundary_tags, metadata=metadata
        )
        dx = ufl.Measure("dx", domain=self._mesh, metadata=metadata)
        lhs = ufl.inner(ufl.grad(test_function), P_2D) * dx
        rhs = ufl.inner(test_function, self._volume_force) * dx

        # Apply boundary confitions
        boundary_conditions = self._domain.define_boundary_conditions(
            function_space=self._function_space,
            measure=ds,
            test_function=test_function,
        )
        dirichlet_bcs, rhs = apply_boundary_conditions(boundary_conditions, rhs)

        # Define problem
        residual_form = lhs - rhs
        problem = NonlinearProblem(residual_form, solution_function, dirichlet_bcs)
        return problem, solution_function

    def _print_maximum_infinitesiaml_strain(self, solution_function: DFunction) -> None:
        strain_function = compute_infinitesimal_strain_function(
            solution_function, self._mesh
        )
        geometric_dim = self._mesh.geometry.dim
        strain = strain_function.x.array.reshape((-1, geometric_dim, geometric_dim))
        max_strain_xx = np.amax(np.absolute(strain[:, 0, 0]))
        max_strain_yy = np.amax(np.absolute(strain[:, 1, 1]))
        max_strain_xy = np.amax(np.absolute(strain[:, 0, 1]))
        max_strain_yx = np.amax(np.absolute(strain[:, 1, 0]))
        print(
            f"Maximum infinitesimal strains: eps_xx = {max_strain_xx}, eps_yy = {max_strain_yy}, eps_xy = {max_strain_xy}, eps_yx = {max_strain_yx}"
        )

    def _print_maximum_green_strain(self, solution_function: DFunction) -> None:
        strain_function = compute_green_strain_function(solution_function, self._mesh)
        geometric_dim = self._mesh.geometry.dim
        strain = strain_function.x.array.reshape((-1, geometric_dim, geometric_dim))
        max_strain_xx = np.amax(np.absolute(strain[:, 0, 0]))
        max_strain_yy = np.amax(np.absolute(strain[:, 1, 1]))
        max_strain_xy = np.amax(np.absolute(strain[:, 0, 1]))
        max_strain_yx = np.amax(np.absolute(strain[:, 1, 0]))
        print(
            f"Maximum Green strains: E_xx = {max_strain_xx}, E_yy = {max_strain_yy}, E_xy = {max_strain_xy}, E_yx = {max_strain_yx}"
        )
