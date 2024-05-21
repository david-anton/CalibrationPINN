from dataclasses import dataclass
from typing import TypeAlias, Union

from dolfinx import default_scalar_type
from dolfinx.fem import Constant, functionspace

from calibrationpinn.fem.base import (
    DFunction,
    DMesh,
    DMeshTags,
    join_simulation_output_subdir,
)
from calibrationpinn.fem.domains import (
    Domain,
    DomainConfig,
    create_domain,
    save_boundary_tags_as_xdmf,
)
from calibrationpinn.fem.problems import (
    LinearElasticityProblemConfig_E_nu,
    LinearElasticityProblemConfig_K_G,
    NeoHookeProblemConfig,
    Problem,
    ProblemConfigs,
    SimulationResults,
    define_problem,
    save_results_as_xdmf,
)
from calibrationpinn.io import ProjectDirectory
from calibrationpinn.io.readerswriters import DataclassWriter

ProblemConfigList: TypeAlias = Union[
    list[LinearElasticityProblemConfig_E_nu],
    list[LinearElasticityProblemConfig_K_G],
    list[NeoHookeProblemConfig],
]


@dataclass
class SimulationConfig:
    domain_config: DomainConfig
    problem_config: ProblemConfigs
    volume_force_x: float
    volume_force_y: float


def run_simulation(
    simulation_config: SimulationConfig,
    save_results: bool,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> SimulationResults:
    domain = create_domain(
        domain_config=simulation_config.domain_config,
        save_mesh=save_results,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )
    simulation_results = _simulate_once(
        simulation_config=simulation_config,
        domain=domain,
        save_results=save_results,
        save_metadata=save_metadata,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )
    return simulation_results


def generate_simulation_data(
    domain_config: DomainConfig,
    problem_configs: ProblemConfigList,
    volume_force_x: float,
    volume_force_y: float,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir=True,
) -> None:
    save_results = True
    save_to_input_dir = save_to_input_dir
    num_simulations = len(problem_configs)
    domain = create_domain(
        domain_config=domain_config,
        save_mesh=save_results,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )
    for simulation_count, problem_config in enumerate(problem_configs):
        print(f"Run FEM simulation {simulation_count + 1}/{num_simulations} ...")
        simulation_config = SimulationConfig(
            domain_config=domain_config,
            problem_config=problem_config,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
        )
        simulation_name = f"sample_{simulation_count}"
        simulation_output_subdir = join_simulation_output_subdir(
            simulation_name, output_subdir
        )
        simulation_results = _simulate_once(
            simulation_config=simulation_config,
            domain=domain,
            save_results=save_results,
            save_metadata=save_metadata,
            output_subdir=simulation_output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )


def _simulate_once(
    simulation_config: SimulationConfig,
    domain: Domain,
    save_results: bool,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> SimulationResults:
    mesh = domain.mesh
    problem_config = simulation_config.problem_config
    volume_force_x = simulation_config.volume_force_x
    volume_force_y = simulation_config.volume_force_y
    element_family = simulation_config.domain_config.element_family
    element_degree = simulation_config.domain_config.element_degree

    volume_force = Constant(
        mesh, (default_scalar_type((volume_force_x, volume_force_y)))
    )

    element_shape = (2,)
    element = (element_family, element_degree, element_shape)
    function_space = functionspace(mesh, element)

    problem = define_problem(
        problem_config=problem_config,
        domain=domain,
        function_space=function_space,
        volume_force=volume_force,
    )

    approximate_solution = problem.solve()
    results = problem.compile_results(approximate_solution=approximate_solution)

    if save_results:
        _save_simulation_results(
            simulation_config=simulation_config,
            results=results,
            approximate_solution=approximate_solution,
            mesh=mesh,
            problem=problem,
            boundary_tags=domain.boundary_tags,
            save_metadata=save_metadata,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )

    return results


def _save_simulation_results(
    simulation_config: SimulationConfig,
    results: SimulationResults,
    approximate_solution: DFunction,
    mesh: DMesh,
    problem: Problem,
    boundary_tags: DMeshTags,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool,
) -> None:
    problem.save_results(
        results=results,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )
    save_simulation_config(
        simulation_config=simulation_config,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )

    if save_metadata:
        save_boundary_tags_as_xdmf(
            boundary_tags=boundary_tags,
            mesh=mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
        save_results_as_xdmf(
            mesh=mesh,
            approximate_solution=approximate_solution,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )


def save_simulation_config(
    simulation_config: SimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> None:
    data_writer = DataclassWriter(project_directory)
    file_name = "simulation_config"
    data_writer.write(simulation_config, file_name, output_subdir, save_to_input_dir)
