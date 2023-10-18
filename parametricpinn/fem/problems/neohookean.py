from dataclasses import dataclass
from typing import Callable, TypeAlias

import dolfinx
import dolfinx.fem as fem
import numpy as np
import pandas as pd
import ufl
from dolfinx import nls
from petsc4py import PETSc

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
    apply_boundary_conditions,
    save_displacements,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter

UFLSigmaFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]


@dataclass
class NeoHookeanProblemConfig:
    youngs_modulus: float
    poissons_ratio: float


@dataclass
class NeoHookeanResults(BaseSimulationResults):
    youngs_modulus: float
    poissons_ratio: float


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
        solver = nls.petsc.NewtonSolver(self._mesh.comm, self._problem)

        # Set Newton solver options
        solver.atol = 1e-8
        solver.rtol = 1e-8
        solver.convergence_criterion = "incremental"

        solver.report = True
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

        num_iterations, converged = solver.solve(self._solution_function)
        assert converged
        print(f"Number of interations: {num_iterations}")

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
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            youngs_modulus=self._config.youngs_modulus,
            poissons_ratio=self._config.poissons_ratio,
            displacements_x=displacements_x,
            displacements_y=displacements_y,
            function=approximate_solution,
        )

        return results

    def save_results(
        self,
        results: NeoHookeanResults,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ) -> None:
        save_displacements(results, output_subdir, save_to_input_dir, project_directory)
        self._save_parameters(
            results, output_subdir, save_to_input_dir, project_directory
        )

    def _define(self) -> tuple[PETSNonlinearProblem, DFunction]:
        # Solution and test function
        test_function = ufl.TestFunction(self._function_space)
        solution_function = fem.Function(self._function_space)

        # Identity tensor
        spatial_dim = len(solution_function)
        I = ufl.variable(ufl.Identity(spatial_dim))

        # Deformation tensors
        F = ufl.variable(I + ufl.grad(solution_function))  # Deformation gradient
        C = ufl.variable(F.T * F)  # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        I_c = ufl.variable(ufl.tr(C))
        J = ufl.variable(ufl.det(F))

        # Material parameters
        E = PETSc.ScalarType(self._config.youngs_modulus)
        nu = PETSc.ScalarType(self._config.poissons_ratio)
        mu_ = fem.Constant(self._mesh, E / (2 * (1 + nu)))
        lambda_ = fem.Constant(self._mesh, (E * nu) / ((1 + nu) * (1 - 2 * nu)))

        # Strain energy
        psi = (mu_ / 2) * (I_c - 2 - 2 * ufl.ln(J)) + (lambda_ / 2) * (J - 1) ** 2

        # Stress (1. Piola-Kirchoff stress tensor)
        P = ufl.diff(psi, F)

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
        problem = fem.petsc.NonlinearProblem(
            residual_form, solution_function, dirichlet_bcs
        )
        return problem, solution_function

    def _save_parameters(
        self,
        simulation_results: NeoHookeanResults,
        output_subdir: str,
        save_to_input_dir: bool,
        project_directory: ProjectDirectory,
    ) -> None:
        data_writer = PandasDataWriter(project_directory)
        file_name = "parameters"
        results = simulation_results
        results_dict = {
            "youngs_modulus": np.array([results.youngs_modulus]),
            "poissons_ratio": np.array([results.poissons_ratio]),
        }
        results_dataframe = pd.DataFrame(results_dict)
        data_writer.write(
            results_dataframe,
            file_name,
            output_subdir,
            header=True,
            save_to_input_dir=save_to_input_dir,
        )
