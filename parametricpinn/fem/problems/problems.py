from typing import Protocol, TypeAlias

from parametricpinn.errors import FEMProblemConfigError
from parametricpinn.fem.base import (
    DConstant,
    DFunction,
    DMesh,
    UFLTestFunction,
    UFLTrialFunction,
)
from parametricpinn.fem.boundaryconditions import BoundaryConditions
from parametricpinn.fem.problems.linearelasticity import (
    LinearElasticityModel,
    LinearElasticityProblem,
    LinearElasticityResults,
)
from parametricpinn.io import ProjectDirectory

MaterialModel: TypeAlias = LinearElasticityModel
SimulationResults: TypeAlias = LinearElasticityResults


class Problem(Protocol):
    def solve(self) -> DFunction:
        pass

    def compile_results(
        self,
        mesh: DMesh,
        approximate_solution: DFunction,
        material_model: MaterialModel,
    ) -> SimulationResults:
        pass

    def save_results(
        self,
        results: SimulationResults,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ) -> None:
        pass


def define_problem(
    material_model: MaterialModel,
    trial_function: UFLTrialFunction,
    test_function: UFLTestFunction,
    volume_force: DConstant,
    boundary_conditions: BoundaryConditions,
) -> Problem:
    if isinstance(material_model, LinearElasticityModel):
        return LinearElasticityProblem(
            material_model,
            trial_function,
            test_function,
            volume_force,
            boundary_conditions,
        )
    else:
        raise FEMProblemConfigError(
            f"There is no implementation for the requested FEM problem {material_model}."
        )
