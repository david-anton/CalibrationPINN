from dataclasses import dataclass
from typing import Callable, TypeAlias

import dolfinx
import dolfinx.fem as fem
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
class NeoHookeanProblemConfig:
    material_parameter_names = ("bulk modulus", "Rivlin-Saunders c_10")
    material_parameters: MaterialParameters


@dataclass
class NeoHookeanResults(BaseSimulationResults):
    pass


class NeoHookeanProblem:
    def __init__(
        self,
        config: NeoHookeanProblemConfig,
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
        solver.convergence_criterion = "incremental"

        solver.report = True
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

        num_iterations, converged = solver.solve(self._solution_function)
        assert converged
        print(f"Number of iterations: {num_iterations}")

        return self._solution_function

    def compile_results(self, approximate_solution: DFunction) -> NeoHookeanResults:
        coordinates = self._function_space.tabulate_dof_coordinates()
        coordinates_x = coordinates[:, 0].reshape((-1, 1))
        coordinates_y = coordinates[:, 1].reshape((-1, 1))

        displacements = approximate_solution.x.array.reshape(
            (-1, self._mesh.geometry.dim)
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))

        results = NeoHookeanResults(
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
        # Solution and test function
        test_function = ufl.TestFunction(self._function_space)
        solution_function = fem.Function(self._function_space)

        # Identity tensor
        spatial_dim = len(solution_function)
        I = ufl.variable(ufl.Identity(spatial_dim))

        # Deformation gradient
        F = ufl.variable(I + ufl.grad(solution_function))
        J = ufl.variable(ufl.det(F))

        # Unimodular deformation tensors
        uni_F = ufl.variable((J ** (-1 / 3)) * F)  # Unimodular deformation gradient
        uni_C = ufl.variable(uni_F.T * uni_F)  # Unimodular right Cauchy-Green tensor

        # Invariants of unimodular deformation tensors
        uni_I_c = ufl.variable(ufl.tr(uni_C))

        # Material parameters
        K = default_scalar_type(self._config.material_parameters[0])
        c_10 = default_scalar_type(self._config.material_parameters[1])

        # 2. Piola-Kirchoff stress tensor
        inv_uni_C = ufl.variable(ufl.inv(uni_C))
        T = J * K * (J - 1) * inv_uni_C + 2 * J ** (-2 / 3) * (
            c_10 * I - 1 / 3 * c_10 * uni_I_c * inv_uni_C
        )

        # 1. Piola-Kirchoff stress tensor
        P = F * T

        # Define variational form
        metadata = {"quadrature_degree": self._quadrature_degree}
        boundary_tags = self._domain.boundary_tags
        ds = ufl.Measure(
            "ds", domain=self._mesh, subdomain_data=boundary_tags, metadata=metadata
        )
        dx = ufl.Measure("dx", domain=self._mesh, metadata=metadata)
        lhs = ufl.inner(ufl.grad(test_function), P) * dx
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
