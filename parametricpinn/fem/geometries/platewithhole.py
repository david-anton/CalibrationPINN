from dataclasses import dataclass
from typing import Any, Callable, TypeAlias, Union

import dolfinx
import gmsh
import numpy as np
import numpy.typing as npt
import petsc4py
import ufl
from dolfinx.fem import Constant
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import Mesh, locate_entities_boundary, meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import TrialFunction

from parametricpinn.fem.base import DFunctionSpace, DMesh, DMeshTags, GGeometry, GMesh
from parametricpinn.fem.boundaryconditions import (
    BoundaryConditions,
    DirichletBC,
    NeumannBC,
)
from parametricpinn.fem.geometries.base import load_mesh_from_gmsh_model, save_gmesh
from parametricpinn.io import ProjectDirectory


@dataclass
class PlateWithHoleGeometryConfig:
    edge_length: float
    radius: float
    traction_left_x: float
    traction_left_y: float
    element_family: str
    element_degree: int
    mesh_resolution: float

Config: TypeAlias = PlateWithHoleGeometryConfig



class PlateWithHoleGeometry():
    def __init__(self, config: Config):
        self.config = config
        self._geometric_dim = 2
        self._u_x_right = ScalarType(0.0)
        self._u_y_bottom = ScalarType(0.0)
        self._tag_right = 0
        self._tag_bottom = 1
        self._tag_left = 2

    def generate_mesh(
        self,
        save_mesh: bool,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ) -> DMesh:
        print("Generate FEM mesh ...")
        gmsh.initialize()
        gmesh = self._generate_gmesh()
        if save_mesh:
            save_gmesh(output_subdir, save_to_input_dir, project_directory)
        mesh = load_mesh_from_gmsh_model(gmesh, self._geometric_dim)
        gmsh.finalize()
        return mesh

    def tag_boundaries(self, mesh: DMesh) -> DMeshTags:
        locate_right_facet = lambda x: np.isclose(x[0], 0.0)
        locate_bottom_facet = lambda x: np.isclose(x[1], 0.0)
        locate_left_facet = lambda x: np.logical_and(np.isclose(x[0], -self.config.edge_length), x[1] > 0.0)
        boundaries = [
            (self._tag_right, locate_right_facet),
            (self._tag_bottom, locate_bottom_facet),
            (self._tag_left, locate_left_facet),
        ]

        facet_indices_list: list[npt.NDArray[np.int32]] = []
        facet_tags_list: list[npt.NDArray[np.int32]] = []
        for tag, locator_func in boundaries:
            located_facet_indices = locate_entities_boundary(
                mesh, bc_facets_dim, locator_func
            )
            facet_indices_list.append(located_facet_indices)
            facet_tags_list.append(np.full_like(located_facet_indices, tag))
        facet_indices = np.hstack(facet_indices_list).astype(np.int32)
        facet_tags = np.hstack(facet_tags_list).astype(np.int32)
        sorted_facet_indices = np.argsort(facet_indices)
        return meshtags(
            mesh,
            bc_facets_dim,
            facet_indices[sorted_facet_indices],
            facet_tags[sorted_facet_indices],
        )


    def define_boundary_conditions(self, mesh: DMesh, boundary_tags: DMeshTags, func_space: DFunctionSpace) -> BoundaryConditions:
        traction_left = Constant(mesh, (ScalarType((self.config.traction_left_x, self.config.traction_left_y))))
        return [
            DirichletBC(
                tag=self._tag_right,
                value=self._u_x_right,
                dim=0,
                func_space=func_space,
                boundary_tags=boundary_tags,
                bc_facets_dim=bc_facets_dim,
            ),
            DirichletBC(
                tag=self._tag_bottom,
                value=self._u_y_bottom,
                dim=1,
                func_space=func_space,
                boundary_tags=boundary_tags,
                bc_facets_dim=bc_facets_dim,
            ),
            NeumannBC(tag=self._tag_left, value=traction_left, measure=ds, test_func=w),
        ]

    def _generate_gmesh(self) -> GMesh:
        length = self.config.edge_length
        radius = self.config.radius
        resolution = self.config.mesh_resolution
        geometry_kernel = gmsh.model.occ
        solid_marker = 1

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
            gmsh.model.mesh.generate(self._geometric_dim)

        geometry = create_geometry()
        tag_physical_enteties(geometry)
        configure_mesh()
        generate_mesh()

        return gmsh.model
    
    def _get_bc_facets_dim(self, mesh: DMesh) ->