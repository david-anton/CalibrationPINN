from typing import Protocol, TypeAlias, Union

from parametricpinn.errors import FEMProblemConfigError
from parametricpinn.fem.base import DConstant, DFunction, DFunctionSpace
from parametricpinn.fem.domains import Domain
from parametricpinn.fem.problems.base import BaseSimulationResults
from parametricpinn.fem.problems.linearelasticity import (
    LinearElasticityProblem_E_nu,
    LinearElasticityProblem_K_G,
    LinearElasticityProblemConfig,
    LinearElasticityProblemConfig_E_nu,
    LinearElasticityProblemConfig_K_G,
    LinearElasticityResults,
)
from parametricpinn.fem.problems.neohooke import (
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
        results: BaseSimulationResults,
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
    if isinstance(problem_config, LinearElasticityProblemConfig_E_nu):
        return LinearElasticityProblem_E_nu(
            config=problem_config,
            domain=domain,
            function_space=function_space,
            volume_force=volume_force,
        )
    elif isinstance(problem_config, LinearElasticityProblemConfig_K_G):
        return LinearElasticityProblem_K_G(
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
