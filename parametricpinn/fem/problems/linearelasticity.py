from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
import pandas as pd
import ufl
from dolfinx.fem.petsc import LinearProblem

from parametricpinn.errors import FEMConfigurationError
from parametricpinn.fem.base import (
    DConstant,
    DFunction,
    DFunctionSpace,
    PETSLinearProblem,
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
class LinearElasticityProblemConfig:
    model: str
    youngs_modulus: float
    poissons_ratio: float


@dataclass
class LinearElasticityResults(BaseSimulationResults):
    youngs_modulus: float
    poissons_ratio: float


class LinearElasticityProblem:
    def __init__(
        self,
        config: LinearElasticityProblemConfig,
        domain: Domain,
        function_space: DFunctionSpace,
        volume_force: DConstant,
    ):
        self._config = config
        self._domain = domain
        self._mesh = domain.mesh
        self._function_space = function_space
        self._volume_force = volume_force
        self._problem = self._define()

    def solve(self) -> DFunction:
        solution_function = self._problem.solve()
        return solution_function

    def compile_results(
        self, approximate_solution: DFunction
    ) -> LinearElasticityResults:
        coordinates = self._function_space.tabulate_dof_coordinates()
        coordinates_x = coordinates[:, 0].reshape((-1, 1))
        coordinates_y = coordinates[:, 1].reshape((-1, 1))

        displacements = approximate_solution.x.array.reshape(
            (-1, self._mesh.geometry.dim)
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))

        results = LinearElasticityResults(
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            youngs_modulus=self._config.youngs_modulus,
            poissons_ratio=self._config.poissons_ratio,
            displacements_x=displacements_x,
            displacements_y=displacements_y,
            function=approximate_solution
        )

        return results

    def save_results(
        self,
        results: LinearElasticityResults,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ) -> None:
        save_displacements(results, output_subdir, save_to_input_dir, project_directory)
        self._save_parameters(
            results, output_subdir, save_to_input_dir, project_directory
        )

    def _define(self) -> PETSLinearProblem:
        # Trial and test function
        trial_function = ufl.TrialFunction(self._function_space)
        test_function = ufl.TestFunction(self._function_space)

        # Define variational form
        boundary_tags = self._domain.boundary_tags
        ds = ufl.Measure("ds", domain=self._mesh, subdomain_data=boundary_tags)
        dx = ufl.Measure("dx", domain=self._mesh)
        sigma, epsilon = self._sigma_and_epsilon_factory()
        lhs = ufl.inner(epsilon(test_function), sigma(trial_function)) * dx
        rhs = ufl.dot(test_function, self._volume_force) * dx

        # Apply boundary confitions
        boundary_conditions = self._domain.define_boundary_conditions(
            function_space=self._function_space,
            measure=ds,
            test_function=test_function,
        )
        dirichlet_bcs, rhs = apply_boundary_conditions(boundary_conditions, rhs)

        # Define problem
        problem = LinearProblem(
            lhs,
            rhs,
            bcs=dirichlet_bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        return problem

    def _sigma_and_epsilon_factory(self) -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        model = self._config.model
        youngs_modulus = self._config.youngs_modulus
        poissons_ratio = self._config.poissons_ratio
        compliance_matrix = None
        if model == "plane stress":
            compliance_matrix = (1 / youngs_modulus) * ufl.as_matrix(
                [
                    [1.0, -poissons_ratio, 0.0],
                    [-poissons_ratio, 1.0, 0.0],
                    [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
                ]
            )
        elif model == "plane strain":
            compliance_matrix = (1 / youngs_modulus) * ufl.as_matrix(
                [
                    [
                        1.0 - poissons_ratio**2,
                        -poissons_ratio * (1.0 + poissons_ratio),
                        0.0,
                    ],
                    [
                        -poissons_ratio * (1.0 + poissons_ratio),
                        1.0 - poissons_ratio**2,
                        0.0,
                    ],
                    [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
                ]
            )
        else:
            raise FEMConfigurationError(f"Unknown model: {model}")

        elasticity_matrix = ufl.inv(compliance_matrix)

        def sigma(u: UFLTrialFunction) -> UFLOperator:
            return _sigma_voigt_to_matrix(
                ufl.dot(elasticity_matrix, _epsilon_matrix_to_voigt(epsilon(u)))
            )

        def _epsilon_matrix_to_voigt(eps: UFLOperator) -> UFLOperator:
            return ufl.as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

        def _sigma_voigt_to_matrix(sig: UFLOperator) -> UFLOperator:
            return ufl.as_tensor([[sig[0], sig[2]], [sig[2], sig[1]]])

        def epsilon(u: UFLTrialFunction) -> UFLOperator:
            return 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

        return sigma, epsilon

    def _save_parameters(
        self,
        simulation_results: LinearElasticityResults,
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
