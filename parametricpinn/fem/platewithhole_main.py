import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from dolfinx.fem import Constant, FunctionSpace
from petsc4py.PETSc import ScalarType
from ufl import Measure, TestFunction, TrialFunction, VectorElement

from parametricpinn.fem.base import DFunction, DMesh, DMeshTags, join_output_path
from parametricpinn.fem.geometries import (
    Geometry,
    GeometryConfig,
    create_geometry,
    save_results_as_xdmf,
)
from parametricpinn.fem.problems import MaterialModel, SimulationResults, define_problem
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import DataclassWriter, PandasDataWriter


@dataclass
class SimulationConfig:
    geometry_config: GeometryConfig
    material_model: MaterialModel
    volume_force_x: float
    volume_force_y: float
    element_family: str = "Lagrange"
    element_degree: int = 1
    mesh_resolution: float = 1


def run_simulation(
    simulation_config: SimulationConfig,
    save_results: bool,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> SimulationResults:
    geometry = create_geometry(simulation_config.geometry_config)
    mesh = geometry.generate_mesh(
        save_mesh=save_results,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )
    simulation_results = _simulate_once(
        simulation_config=simulation_config,
        geometry=geometry,
        mesh=mesh,
        save_results=save_results,
        save_metadata=save_metadata,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )
    return simulation_results


def generate_validation_data(
    geometry_config: GeometryConfig,
    material_models: list[MaterialModel],
    volume_force_x: float,
    volume_force_y: float,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    element_family: str = "Lagrange",
    element_degree: int = 1,
    mesh_resolution: float = 1,
) -> None:
    save_results = True
    save_to_input_dir = True
    num_simulations = len(material_models)
    geometry = create_geometry(geometry_config)
    mesh = geometry.generate_mesh(
        save_mesh=save_results,
        output_subdir=output_subdir,
        project_directory=project_directory,
        save_to_input_dir=save_to_input_dir,
    )
    for simulation_count, material_model in enumerate(material_models):
        print(f"Run FEM simulation {simulation_count + 1}/{num_simulations} ...")
        simulation_config = SimulationConfig(
            geometry_config=geometry_config,
            material_model=material_model,
            volume_force_x=volume_force_x,
            volume_force_y=volume_force_y,
            element_family=element_family,
            element_degree=element_degree,
            mesh_resolution=mesh_resolution,
        )
        simulation_name = f"sample_{simulation_count}"
        simulation_output_subdir = _join_simulation_output_subdir(
            simulation_name, output_subdir
        )
        simulation_results = _simulate_once(
            simulation_config=simulation_config,
            geometry=geometry,
            mesh=mesh,
            save_results=save_results,
            save_metadata=save_metadata,
            output_subdir=simulation_output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )


def _join_simulation_output_subdir(simulation_name: str, output_subdir: str) -> str:
    return os.path.join(output_subdir, simulation_name)


def _simulate_once(
    simulation_config: SimulationConfig,
    geometry: Geometry,
    mesh: DMesh,
    save_results: bool,
    save_metadata: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> SimulationResults:
    material_model = simulation_config.material_model
    volume_force_x = simulation_config.volume_force_x
    volume_force_y = simulation_config.volume_force_y
    element_family = simulation_config.element_family
    element_degree = simulation_config.element_degree

    volume_force = Constant(mesh, (ScalarType((volume_force_x, volume_force_y))))

    def save_boundary_tags_as_xdmf(boundary_tags: DMeshTags) -> None:
        file_name = "boundary_tags.xdmf"
        output_path = join_output_path(
            project_directory, file_name, output_subdir, save_to_input_dir
        )
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        with XDMFFile(mesh.comm, output_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(boundary_tags)

    def save_results_as_xdmf(mesh: DMesh, uh: DFunction) -> None:
        file_name = "displacements.xdmf"
        output_path = join_output_path(
            project_directory, file_name, output_subdir, save_to_input_dir
        )
        with XDMFFile(mesh.comm, output_path, "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(uh)

    def compile_output(mesh: DMesh, uh: DFunction) -> SimulationResults:
        coordinates = mesh.geometry.x
        coordinates_x = coordinates[:, 0].reshape((-1, 1))
        coordinates_y = coordinates[:, 1].reshape((-1, 1))

        displacements = uh.x.array.reshape((-1, mesh.geometry.dim))
        displacements_x = displacements[:, 0].reshape((-1, 1))
        displacements_y = displacements[:, 1].reshape((-1, 1))

        simulation_results = SimulationResults(
            coordinates_x=coordinates_x,
            coordinates_y=coordinates_y,
            youngs_modulus=youngs_modulus,
            poissons_ratio=poissons_ratio,
            displacements_x=displacements_x,
            displacements_y=displacements_y,
        )

        return simulation_results

    element = VectorElement(element_family, mesh.ufl_cell(), element_degree)
    func_space = FunctionSpace(mesh, element)
    trial_function = TrialFunction(func_space)
    test_function = TestFunction(func_space)

    boundary_tags = geometry.tag_boundaries(mesh)
    ds_measure = Measure("ds", domain=mesh, subdomain_data=boundary_tags)

    boundary_conditions = geometry.define_boundary_conditions(
        mesh=mesh,
        boundary_tags=boundary_tags,
        function_space=func_space,
        measure=ds_measure,
        test_function=test_function,
    )

    problem = define_problem(
        material_model=material_model,
        trial_function=trial_function,
        test_function=test_function,
        volume_force=volume_force,
        boundary_conditions=boundary_conditions,
    )

    uh = problem.solve()

    if save_metadata:
        save_boundary_tags_as_xdmf(boundary_tags)
        save_results_as_xdmf(mesh, uh)

    return compile_output(mesh, uh)


def _save_results(
    simulation_results: SimulationResults,
    simulation_config: PWHSimulationConfig,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> None:
    _save_simulation_results(
        simulation_results, output_subdir, save_to_input_dir, project_directory
    )
    _save_simulation_config(
        simulation_config, output_subdir, save_to_input_dir, project_directory
    )


def _save_simulation_results(
    simulation_results: SimulationResults,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    _save_displacements(
        simulation_results, output_subdir, save_to_input_dir, project_directory
    )
    _save_parameters(
        simulation_results, output_subdir, save_to_input_dir, project_directory
    )


def _save_displacements(
    simulation_results: SimulationResults,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    data_writer = PandasDataWriter(project_directory)
    file_name = "displacements"
    results = simulation_results
    results_dict = {
        "coordinates_x": np.ravel(results.coordinates_x),
        "coordinates_y": np.ravel(results.coordinates_y),
        "displacements_x": np.ravel(results.displacements_x),
        "displacements_y": np.ravel(results.displacements_y),
    }
    results_dataframe = pd.DataFrame(results_dict)
    data_writer.write(
        results_dataframe,
        file_name,
        output_subdir,
        header=True,
        save_to_input_dir=save_to_input_dir,
    )


def _save_parameters(
    simulation_results: SimulationResults,
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


def _save_simulation_config(
    simulation_config: PWHSimulationConfig,
    output_subdir: str,
    save_to_input_dir: bool,
    project_directory: ProjectDirectory,
) -> None:
    data_writer = DataclassWriter(project_directory)
    file_name = "simulation_config"
    data_writer.write(
        simulation_config, file_name, output_subdir, save_to_input_dir=save_to_input_dir
    )
