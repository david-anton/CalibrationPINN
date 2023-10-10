from typing import Protocol, TypeAlias, Union

from parametricpinn.errors import FEMProblemConfigError
from parametricpinn.fem.base import DConstant, DFunction, DFunctionSpace
from parametricpinn.fem.domains import Domain
from parametricpinn.fem.problems.linearelasticity import (
    LinearElasticityProblem,
    LinearElasticityProblemConfig,
    LinearElasticityResults,
)
from parametricpinn.fem.problems.neohookean import (
    NeoHookeanProblem,
    NeoHookeanProblemConfig,
    NeoHookeanResults,
)
from parametricpinn.io import ProjectDirectory

ProblemConfigs: TypeAlias = Union[
    LinearElasticityProblemConfig, NeoHookeanProblemConfig
]
SimulationResults: TypeAlias = Union[LinearElasticityResults, NeoHookeanResults]


class Problem(Protocol):
    def solve(self) -> DFunction:
        pass

    def compile_results(self, approximate_solution: DFunction) -> SimulationResults:
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
    problem_config: ProblemConfigs,
    domain: Domain,
    function_space: DFunctionSpace,
    volume_force: DConstant,
) -> Problem:
    if isinstance(problem_config, LinearElasticityProblemConfig):
        return LinearElasticityProblem(
            config=problem_config,
            domain=domain,
            function_space=function_space,
            volume_force=volume_force,
        )
    elif isinstance(problem_config, NeoHookeanProblemConfig):
        return NeoHookeanProblem(
            config=problem_config,
            domain=domain,
            function_space=function_space,
            volume_force=volume_force,
        )
    else:
        raise FEMProblemConfigError(
            f"There is no implementation for the requested FEM problem {problem_config}."
        )
