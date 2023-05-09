import dolfinx
from dataclasses import dataclass
import gmsh
import sys
from typing import Optional
from parametricpinn.io import ProjectDirectory
from parametricpinn.settings import Settings


@dataclass
class PWHSimulationConfig:
    """Configuration class for the FEM simulation of a plate with a hole."""

    edge_length: float
    radius: float
    element_family: str = "Lagrange"
    element_degree: int = 1
    resolution: float = 1


def run_simulation(
    config: PWHSimulationConfig, output_subdir: str, project_directory: ProjectDirectory
) -> None:
    mesh = _generate_mesh(config, output_subdir, project_directory)


def _generate_mesh(
    config: PWHSimulationConfig, output_subdir: str, project_directory: ProjectDirectory
) -> gmsh.model:
    length = config.edge_length
    radius = config.radius
    resolution = config.resolution
    geometry_kernel = gmsh.model.occ  # Use open CASCADE as geometry kernel

    def create_geometry() -> None:
        gmsh.model.add("domain")
        plate = geometry_kernel.add_rectangle(0, 0, 0, -length, length)
        hole = geometry_kernel.add_disk(0, 0, 0, radius, radius)
        return geometry_kernel.cut([(2, plate)], [(2, hole)])

    def mesh_size_callback(
        dim: int, tag: int, x: float, y: float, z: float, lc: float
    ) -> float:
        return resolution

    def save_mesh() -> None:
        output_path = project_directory.create_output_file_path(
            file_name="mesh.msh", subdir_name=output_subdir
        )
        gmsh.write(str(output_path))

    gmsh.initialize()
    create_geometry()
    gmsh.model.mesh.setSizeCallback(mesh_size_callback)
    geometry_kernel.synchronize()
    gmsh.model.mesh.generate(2)
    save_mesh()
    gmsh.finalize()

    return gmsh.model


if __name__ == "__main__":
    settings = Settings()
    project_directory = ProjectDirectory(settings)
    simulation_config = PWHSimulationConfig(edge_length=100, radius=10)
    run_simulation(
        config=simulation_config,
        output_subdir="mesh_test",
        project_directory=project_directory,
    )

    # surface_geometry = geometry_kernel.getEntities(dim=2)
    # assert surface_geometry == geometry[0]
    # surface_geometry = surface_geometry[0]
    # field = gmsh.model.mesh.field
    # constant_meshsize_field = field.add("Constant")
    # field.setNumbers(constant_meshsize_field, "SurfacesList", [surface_geometry[1]])
    # field.setNumber(constant_meshsize_field, "IncludeBoundary", 1)
    # field.setNumber(constant_meshsize_field, "VIn", mesh_resolution)
    # field.setNumber(constant_meshsize_field, "VOut", mesh_resolution)

    # gmsh.option.setNumber("Mesh.MeshSizeMin", 3)
    # gmsh.option.setNumber("Mesh.MeshSizeMax", 3)

    # gmsh.model.mesh.setSize(gmsh.model.getEntities(0), resolution)
