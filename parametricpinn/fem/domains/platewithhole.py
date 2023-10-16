from dataclasses import dataclass
from typing import TypeAlias

import gmsh
import numpy as np
from dolfinx.fem import Constant
from dolfinx.mesh import meshtags
from petsc4py.PETSc import ScalarType

from parametricpinn.fem.base import (
    DFunctionSpace,
    DMesh,
    DMeshTags,
    GGeometry,
    GMesh,
    UFLMeasure,
    UFLTestFunction,
)
from parametricpinn.fem.boundaryconditions import (
    BoundaryConditions,
    DirichletBC,
    NeumannBC,
)
from parametricpinn.fem.domains.base import (
    list_sorted_facet_indices_and_tags,
    load_mesh_from_gmsh_model,
    save_gmesh,
)
from parametricpinn.io import ProjectDirectory


@dataclass
class PlateWithHoleDomainConfig:
    plate_length: float
    plate_height: float
    hole_radius: float
    traction_right_x: float
    traction_right_y: float
    element_family: str
    element_degree: int
    mesh_resolution: float


Config: TypeAlias = PlateWithHoleDomainConfig


class PlateWithHoleDomain:
    def __init__(
        self,
        config: Config,
        save_mesh: bool,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ):
        self.config = config
        self._geometric_dim = 2
        self._u_x_left = ScalarType(0.0)
        self._tag_left = 0
        self._tag_right = 1
        self.mesh = self._generate_mesh(
            save_mesh=save_mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
        self._bc_facets_dim = self.mesh.topology.dim - 1
        self.boundary_tags = self._tag_boundaries()

    def define_boundary_conditions(
        self,
        function_space: DFunctionSpace,
        measure: UFLMeasure,
        test_function: UFLTestFunction,
    ) -> BoundaryConditions:
        traction_right = Constant(
            self.mesh,
            (ScalarType((self.config.traction_right_x, self.config.traction_right_y))),
        )
        return [
            DirichletBC(
                tag=self._tag_left,
                value=self._u_x_left,
                dim=0,
                function_space=function_space,
                boundary_tags=self.boundary_tags,
                bc_facets_dim=self._bc_facets_dim,
            ),
            NeumannBC(
                tag=self._tag_right,
                value=traction_right,
                measure=measure,
                test_function=test_function,
            ),
        ]

    def _generate_mesh(
        self,
        save_mesh: bool,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool,
    ) -> DMesh:
        print("Generate FEM mesh ...")
        gmsh.initialize()
        gmesh = self._generate_gmesh()
        if save_mesh:
            save_gmesh(output_subdir, save_to_input_dir, project_directory)
        mesh = load_mesh_from_gmsh_model(gmesh, self._geometric_dim)
        gmsh.finalize()
        return mesh

    def _generate_gmesh(self) -> GMesh:
        length = self.config.plate_length
        height = self.config.plate_height
        radius = self.config.hole_radius
        resolution = self.config.mesh_resolution
        geometry_kernel = gmsh.model.occ
        solid_marker = 1

        def create_geometry() -> GGeometry:
            gmsh.model.add("domain")
            plate = geometry_kernel.add_rectangle(0, 0, 0, length, height)
            hole = geometry_kernel.add_disk(length / 2, height / 2, 0, radius, radius)
            return geometry_kernel.cut([(2, plate)], [(2, hole)])

        def tag_physical_enteties(geometry: GGeometry) -> None:
            geometry_kernel.synchronize()

            def tag_solid_surface() -> None:
                surface = geometry_kernel.getEntities(dim=2)
                assert surface == geometry[0]
                gmsh.model.addPhysicalGroup(
                    surface[0][0], [surface[0][1]], solid_marker
                )
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

    def _tag_boundaries(self) -> DMeshTags:
        locate_left_facet = lambda x: np.isclose(x[0], 0.0)
        locate_right_facet = lambda x: np.isclose(x[0], self.config.plate_length)
        boundaries = [
            (self._tag_left, locate_left_facet),
            (self._tag_right, locate_right_facet),
        ]
        sorted_facet_indices, sorted_facet_tags = list_sorted_facet_indices_and_tags(
            boundaries=boundaries, mesh=self.mesh, bc_facets_dim=self._bc_facets_dim
        )
        return meshtags(
            self.mesh,
            self._bc_facets_dim,
            sorted_facet_indices,
            sorted_facet_tags,
        )
