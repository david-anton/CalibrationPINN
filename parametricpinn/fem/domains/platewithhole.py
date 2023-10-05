from dataclasses import dataclass
from typing import TypeAlias

import gmsh
import numpy as np
import numpy.typing as npt
from dolfinx.fem import Constant
from dolfinx.mesh import locate_entities_boundary, meshtags
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
from parametricpinn.fem.domains.base import load_mesh_from_gmsh_model, save_gmesh
from parametricpinn.io import ProjectDirectory

FacetsDim: TypeAlias = int


@dataclass
class PlateWithHoleDomainConfig:
    edge_length: float
    radius: float
    traction_left_x: float
    traction_left_y: float
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
        self.mesh = self._generate_mesh(
            save_mesh=save_mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
        self.boundary_tags = self._tag_boundaries()
        self._geometric_dim = 2
        self._bc_facets_dim = self.mesh.topology.dim - 1
        self._u_x_right = ScalarType(0.0)
        self._u_y_bottom = ScalarType(0.0)
        self._tag_right = 0
        self._tag_bottom = 1
        self._tag_left = 2

    def define_boundary_conditions(
        self,
        function_space: DFunctionSpace,
        measure: UFLMeasure,
        test_function: UFLTestFunction,
    ) -> BoundaryConditions:
        traction_left = Constant(
            self.mesh,
            (ScalarType((self.config.traction_left_x, self.config.traction_left_y))),
        )
        return [
            DirichletBC(
                tag=self._tag_right,
                value=self._u_x_right,
                dim=0,
                function_space=function_space,
                boundary_tags=self.boundary_tags,
                bc_facets_dim=self._bc_facets_dim,
            ),
            DirichletBC(
                tag=self._tag_bottom,
                value=self._u_y_bottom,
                dim=1,
                function_space=function_space,
                boundary_tags=self.boundary_tags,
                bc_facets_dim=self._bc_facets_dim,
            ),
            NeumannBC(
                tag=self._tag_left,
                value=traction_left,
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
        locate_right_facet = lambda x: np.isclose(x[0], 0.0)
        locate_bottom_facet = lambda x: np.isclose(x[1], 0.0)
        locate_left_facet = lambda x: np.logical_and(
            np.isclose(x[0], -self.config.edge_length), x[1] > 0.0
        )
        boundaries = [
            (self._tag_right, locate_right_facet),
            (self._tag_bottom, locate_bottom_facet),
            (self._tag_left, locate_left_facet),
        ]

        facet_indices_list: list[npt.NDArray[np.int32]] = []
        facet_tags_list: list[npt.NDArray[np.int32]] = []
        for tag, locator_func in boundaries:
            located_facet_indices = locate_entities_boundary(
                self.mesh, self._bc_facets_dim, locator_func
            )
            facet_indices_list.append(located_facet_indices)
            facet_tags_list.append(np.full_like(located_facet_indices, tag))
        facet_indices = np.hstack(facet_indices_list).astype(np.int32)
        facet_tags = np.hstack(facet_tags_list).astype(np.int32)
        sorted_facet_indices = np.argsort(facet_indices)
        return meshtags(
            self.mesh,
            self._bc_facets_dim,
            facet_indices[sorted_facet_indices],
            facet_tags[sorted_facet_indices],
        )
