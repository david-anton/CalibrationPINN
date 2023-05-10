import dolfinx
from dataclasses import dataclass
import gmsh
import sys
from typing import Optional, Any
from parametricpinn.io import ProjectDirectory
from parametricpinn.settings import Settings
from typing import TypeAlias
from dolfinx.io.gmshio import model_to_mesh
from dolfinx import fem
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import Mesh, locate_entities_boundary
import ufl
from mpi4py import MPI
import numpy as np
from parametricpinn.types import NPArray
from petsc4py.PETSc import ScalarType


@dataclass
class PWHSimulationConfig:
    edge_length: float
    radius: float
    element_family: str = "Lagrange"
    element_degree: int = 1
    resolution: float = 50


Config: TypeAlias = PWHSimulationConfig
GMesh: TypeAlias = gmsh.model
GGeometry: TypeAlias = tuple[
    list[tuple[int, int]], list[list[tuple[int, int]], list[Any]]
]


def run_simulation(
    config: Config, output_subdir: str, project_directory: ProjectDirectory
) -> None:
    gmsh.initialize()
    gmesh = _generate_mesh(config, output_subdir, project_directory)
    _simulate(gmesh, config, output_subdir, project_directory)
    gmsh.finalize()


def _generate_mesh(
    config: Config, output_subdir: str, project_directory: ProjectDirectory
) -> GMesh:
    length = config.edge_length
    radius = config.radius
    resolution = config.resolution
    geometry_kernel = gmsh.model.occ  # Use open CASCADE as geometry kernel

    def create_geometry() -> GGeometry:
        gmsh.model.add("domain")
        plate = geometry_kernel.add_rectangle(0, 0, 0, -length, length)
        hole = geometry_kernel.add_disk(0, 0, 0, radius, radius)
        return geometry_kernel.cut([(2, plate)], [(2, hole)])

    def tag_physical_enteties(geometry: GGeometry) -> None:
        geometry_kernel.synchronize()

        def tag_solid_surface() -> None:
            surface = geometry_kernel.getEntities(dim=2)
            assert surface == geometry[0]
            solid_marker = 1
            gmsh.model.addPhysicalGroup(surface[0][0], [surface[0][1]], solid_marker)
            gmsh.model.setPhysicalName(surface[0][0], solid_marker, "Solid")

        tag_solid_surface()

    def configure_mesh() -> None:
        gmsh.model.mesh.setSizeCallback(mesh_size_callback)

    def mesh_size_callback(
        dim: int, tag: int, x: float, y: float, z: float, lc: float
    ) -> float:
        return resolution

    def generate_mesh() -> None:
        geometry_kernel.synchronize()
        gmsh.model.mesh.generate(2)

    def save_mesh() -> None:
        output_path = project_directory.create_output_file_path(
            file_name="mesh.msh", subdir_name=output_subdir
        )
        gmsh.write(str(output_path))

    geometry = create_geometry()
    tag_physical_enteties(geometry)
    configure_mesh()
    generate_mesh()
    save_mesh()

    return gmsh.model


def _simulate(
    gmesh: GMesh,
    config: Config,
    output_subdir: str,
    project_directory: ProjectDirectory,
) -> None:
    element_family = config.element_family
    element_degree = config.element_degree

    def load_mesh(gmesh: GMesh) -> Mesh:
        rank = 0  # The rank the Gmsh model is initialized on.
        geometrical_dim = 2
        mesh, cell_tags, facet_tags = model_to_mesh(
            gmesh, MPI.COMM_WORLD, rank, gdim=geometrical_dim
        )
        return mesh

    ####

    mesh = load_mesh(gmesh)
    element = ufl.VectorElement(element_family, mesh.ufl_cell(), element_degree)
    func_space = fem.FunctionSpace(mesh, element)

    ####

    def locate_bottom_boundary(x: NPArray) -> bool:
        return np.isclose(x[1], 0)

    def locate_right_boundary(x: NPArray) -> bool:
        return np.isclose(x[0], 0)

    facets_dim = mesh.topology.dim - 1
    bottom_boundary_facets = locate_entities_boundary(
        mesh, facets_dim, locate_bottom_boundary
    )
    bottom_boundary_dofs_y = locate_dofs_topological(
        func_space.sub(1), facets_dim, bottom_boundary_facets
    )
    bottom_D_bc = fem.dirichletbc(
        ScalarType(0),
        bottom_boundary_dofs_y,
        func_space.sub(1),
    )

    right_boundary_facets = locate_entities_boundary(
        mesh, facets_dim, locate_right_boundary
    )
    right_boundary_dofs_x = locate_dofs_topological(
        func_space.sub(0), facets_dim, right_boundary_facets
    )
    right_D_bc = fem.dirichletbc(
        ScalarType(0),
        right_boundary_dofs_x,
        func_space.sub(0),
    )

    bcs = [bottom_D_bc, right_D_bc]

    ###


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
