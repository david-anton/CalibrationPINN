from dataclasses import dataclass
from typing import Callable, TypeAlias

from dolfinx.fem.petsc import LinearProblem
from ufl import (
    TrialFunction,
    as_matrix,
    as_tensor,
    as_vector,
    dot,
    dx,
    inner,
    inv,
    nabla_grad,
)

from parametricpinn.errors import FEMConfigurationError
from parametricpinn.fem.base import (
    DConstant,
    DFunction,
    PETSLinearProblem,
    UFLOperator,
    UFLTestFunction,
    UFLTrialFunction,
)
from parametricpinn.fem.boundaryconditions import BoundaryConditions
from parametricpinn.fem.problems.base import apply_boundary_conditions
from parametricpinn.types import NPArray

UFLSigmaFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]
UFLEpsilonFunc: TypeAlias = Callable[[UFLTrialFunction], UFLOperator]


@dataclass
class LinearElasticityModel:
    model: str
    youngs_modulus: float
    poissons_ratio: float


@dataclass
class LinearElasticityResults:
    coordinates_x: NPArray
    coordinates_y: NPArray
    youngs_modulus: float
    poissons_ratio: float
    displacements_x: NPArray
    displacements_y: NPArray


class LinearElasticityProblem:
    def __init__(
        self,
        config: LinearElasticityModel,
        trial_function: UFLTrialFunction,
        test_function: UFLTestFunction,
        volume_force: DConstant,
        boundary_conditions: BoundaryConditions,
    ):
        self.config = config
        self.problem = self._define(
            trial_function, test_function, volume_force, boundary_conditions
        )

    def solve(self) -> DFunction:
        return self.problem.solve()

    def _define(
        self,
        trial_function: UFLTrialFunction,
        test_function: UFLTestFunction,
        volume_force: DConstant,
        boundary_conditions: BoundaryConditions,
    ) -> PETSLinearProblem:
        sigma, epsilon = self._sigma_and_epsilon_factory()
        lhs = inner(epsilon(test_function), sigma(trial_function)) * dx
        rhs = dot(test_function, volume_force) * dx
        dirichlet_bcs, rhs = apply_boundary_conditions(boundary_conditions, rhs)
        return LinearProblem(
            lhs,
            rhs,
            bcs=dirichlet_bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    def _sigma_and_epsilon_factory(self) -> tuple[UFLSigmaFunc, UFLEpsilonFunc]:
        model = self.config.model
        youngs_modulus = self.config.youngs_modulus
        poissons_ratio = self.config.poissons_ratio
        compliance_matrix = None
        if model == "plane stress":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
                [
                    [1.0, -poissons_ratio, 0.0],
                    [-poissons_ratio, 1.0, 0.0],
                    [0.0, 0.0, 2 * (1.0 + poissons_ratio)],
                ]
            )
        elif model == "plane strain":
            compliance_matrix = (1 / youngs_modulus) * as_matrix(
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

        elasticity_matrix = inv(compliance_matrix)

        def sigma(u: TrialFunction) -> UFLOperator:
            return _sigma_voigt_to_matrix(
                dot(elasticity_matrix, _epsilon_matrix_to_voigt(epsilon(u)))
            )

        def _epsilon_matrix_to_voigt(eps: UFLOperator) -> UFLOperator:
            return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

        def _sigma_voigt_to_matrix(sig: UFLOperator) -> UFLOperator:
            return as_tensor([[sig[0], sig[2]], [sig[2], sig[1]]])

        def epsilon(u: TrialFunction) -> UFLOperator:
            return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

        return sigma, epsilon
