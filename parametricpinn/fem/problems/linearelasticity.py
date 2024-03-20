from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, TypeAlias, Union

import dolfinx
import ufl
from dolfinx import default_scalar_type
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
    MaterialParameters,
    apply_boundary_conditions,
    print_maximum_infinitesiaml_strain,
    save_displacements,
    save_parameters,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.types import NPArray

UFLSigmaFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]


@dataclass
class BaseLinearElasticityProblemConfig:
    model: str
    material_parameters: MaterialParameters


@dataclass
class LinearElasticityProblemConfig_E_nu(BaseLinearElasticityProblemConfig):
    material_parameter_names = ("Youngs modulus", "Poissons ratio")


@dataclass
class LinearElasticityProblemConfig_K_G(BaseLinearElasticityProblemConfig):
    material_parameter_names = ("bulk modulus", "shear modulus")


LinearElasticityProblemConfig: TypeAlias = Union[
    LinearElasticityProblemConfig_E_nu, LinearElasticityProblemConfig_K_G
]


@dataclass
class LinearElasticityResults(BaseSimulationResults):
    pass


class LinearElasticityProblemBase(ABC):
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
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
        solution_function = self._problem.solve()
        print_maximum_infinitesiaml_strain(solution_function, self._mesh)
        return solution_function

    @abstractmethod
    def compile_results(
        self, approximate_solution: DFunction
    ) -> LinearElasticityResults:
        pass

    def save_results(
        self,
        results: BaseSimulationResults,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ) -> None:
        save_displacements(results, output_subdir, save_to_input_dir, project_directory)
        save_parameters(results, output_subdir, save_to_input_dir, project_directory)

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

    @abstractmethod
    def _sigma_and_epsilon_factory(self) -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        pass

    def _extract_coordinates_results(self) -> tuple[NPArray, NPArray]:
        coordinates = self._function_space.tabulate_dof_coordinates()
        coordinates_x = coordinates[:, 0].reshape((-1, 1))
        coordinates_y = coordinates[:, 1].reshape((-1, 1))
        return coordinates_x, coordinates_y

    def _extract_displacements_results(
        self, approximate_solution: DFunction
    ) -> tuple[NPArray, NPArray]:
        displacements = approximate_solution.x.array.reshape(
            (-1, self._mesh.geometry.dim)
        )
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))
        return displacements_x, displacements_y


class LinearElasticityProblem_E_nu(LinearElasticityProblemBase):
    def __init__(
        self,
        config: LinearElasticityProblemConfig_E_nu,
        domain: Domain,
        function_space: DFunctionSpace,
        volume_force: DConstant,
    ):
        super().__init__(config, domain, function_space, volume_force)

    def compile_results(
        self, approximate_solution: DFunction
    ) -> LinearElasticityResults:
        coordinates_x, coordinates_y = self._extract_coordinates_results()
        displacements_x, displacements_y = self._extract_displacements_results(
            approximate_solution
        )

        return LinearElasticityResults(
            material_parameter_names=self._config.material_parameter_names,
            material_parameters=self._config.material_parameters,
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            displacements_x=displacements_x,
            displacements_y=displacements_y,
            function=approximate_solution,
        )

    def _sigma_and_epsilon_factory(self) -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        model = self._config.model
        youngs_modulus = self._config.material_parameters[0]
        poissons_ratio = self._config.material_parameters[1]
        E = default_scalar_type(youngs_modulus)
        nu = default_scalar_type(poissons_ratio)
        compliance_matrix = None
        if model == "plane stress":
            compliance_matrix = (1 / E) * ufl.as_matrix(
                [
                    [1.0, -nu, 0.0],
                    [-nu, 1.0, 0.0],
                    [0.0, 0.0, 2 * (1.0 + nu)],
                ]
            )
        elif model == "plane strain":
            compliance_matrix = (1 / E) * ufl.as_matrix(
                [
                    [
                        1.0 - nu**2,
                        -nu * (1.0 + nu),
                        0.0,
                    ],
                    [
                        -nu * (1.0 + nu),
                        1.0 - nu**2,
                        0.0,
                    ],
                    [0.0, 0.0, 2 * (1.0 + nu)],
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
            # equivalent to 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            return ufl.sym(ufl.grad(u))

        return sigma, epsilon


class LinearElasticityProblem_K_G(LinearElasticityProblemBase):
    def __init__(
        self,
        config: LinearElasticityProblemConfig_K_G,
        domain: Domain,
        function_space: DFunctionSpace,
        volume_force: DConstant,
    ):
        super().__init__(config, domain, function_space, volume_force)

    def compile_results(
        self, approximate_solution: DFunction
    ) -> LinearElasticityResults:
        coordinates_x, coordinates_y = self._extract_coordinates_results()
        displacements_x, displacements_y = self._extract_displacements_results(
            approximate_solution
        )

        return LinearElasticityResults(
            material_parameter_names=self._config.material_parameter_names,
            material_parameters=self._config.material_parameters,
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            displacements_x=displacements_x,
            displacements_y=displacements_y,
            function=approximate_solution,
        )

    def _sigma_and_epsilon_factory(self) -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        model = self._config.model
        bulk_modulus = self._config.material_parameters[0]
        shear_modulus = self._config.material_parameters[1]
        K = default_scalar_type(bulk_modulus)
        G = default_scalar_type(shear_modulus)

        def epsilon(u: UFLTrialFunction) -> UFLOperator:
            # equivalent to 0.5 * (ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
            return ufl.sym(ufl.grad(u))

        if model == "plane stress":

            def sigma(u: UFLTrialFunction) -> UFLOperator:
                strain = epsilon(u)
                eps_xx = strain[0, 0]
                eps_yy = strain[1, 1]
                eps_xy = strain[0, 1]
                sig_xx = (2 * G / (3 * K + 4 * G)) * (
                    (6 * K + 2 * G) * eps_xx + (3 * K - 2 * G) * eps_yy
                )
                sig_yy = (2 * G / (3 * K + 4 * G)) * (
                    (3 * K - 2 * G) * eps_xx + (6 * K + 2 * G) * eps_yy
                )
                sig_xy = G * 2 * eps_xy
                return ufl.as_tensor([[sig_xx, sig_xy], [sig_xy, sig_yy]])

        elif model == "plane strain":

            def volumetric_strain_func(u: UFLTrialFunction) -> UFLOperator:
                spatial_dim = len(u)
                I = ufl.Identity(spatial_dim)
                trace_strain = ufl.tr(epsilon(u))
                return trace_strain * I

            def deviatoric_strain_func(u: UFLTrialFunction) -> UFLOperator:
                strain = epsilon(u)
                volumetric_strain = volumetric_strain_func(u)
                return strain - (volumetric_strain / 2)

            def sigma(u: UFLTrialFunction) -> UFLOperator:
                volumetric_strain = volumetric_strain_func(u)
                deviatoric_strain = deviatoric_strain_func(u)
                return K * volumetric_strain + 2 * G * deviatoric_strain

        else:
            raise FEMConfigurationError(f"Unknown model: {model}")

        return sigma, epsilon
