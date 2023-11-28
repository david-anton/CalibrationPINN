from dataclasses import dataclass
from typing import TypeAlias

import gmsh
import numpy as np
from dolfinx import default_scalar_type
from dolfinx.fem import Constant
from dolfinx.mesh import meshtags

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
    NeumannBC,
    SubDirichletBC,
)
from parametricpinn.fem.domains.base import (
    list_sorted_facet_indices_and_tags,
    load_mesh_from_gmsh_model,
    save_gmesh,
)
from parametricpinn.io import ProjectDirectory


@dataclass
class QuarterPlateWithHoleDomainConfig:
    edge_length: float
    radius: float
    traction_left_x: float
    traction_left_y: float
    element_family: str
    element_degree: int
    element_size: float


class QuarterPlateWithHoleDomain:
    def __init__(
        self,
        config: QuarterPlateWithHoleDomainConfig,
        save_mesh: bool,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ):
        self.config = config
        self._geometric_dim = 2
        self._u_x_right = default_scalar_type(0.0)
        self._u_y_bottom = default_scalar_type(0.0)
        self._tag_left = 0
        self._tag_right = 1
        self._tag_bottom = 2
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
        traction_left = Constant(
            self.mesh,
            (
                default_scalar_type(
                    (self.config.traction_left_x, self.config.traction_left_y)
                )
            ),
        )
        u_x_right = Constant(self.mesh, self._u_x_right)
        u_y_bottom = Constant(self.mesh, self._u_y_bottom)
        return [
            NeumannBC(
                tag=self._tag_left,
                value=traction_left,
                measure=measure,
                test_function=test_function,
            ),
            SubDirichletBC(
                tag=self._tag_right,
                value=u_x_right,
                dim=0,
                function_space=function_space,
                boundary_tags=self.boundary_tags,
                bc_facets_dim=self._bc_facets_dim,
            ),
            SubDirichletBC(
                tag=self._tag_bottom,
                value=u_y_bottom,
                dim=1,
                function_space=function_space,
                boundary_tags=self.boundary_tags,
                bc_facets_dim=self._bc_facets_dim,
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
        element_size = self.config.element_size
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
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size)

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
        # Approximate solution converges much slower when excluding x[1]=0
        locate_left_facet = lambda x: np.isclose(x[0], -self.config.edge_length)
        boundaries = [
            (self._tag_right, locate_right_facet),
            (self._tag_bottom, locate_bottom_facet),
            (self._tag_left, locate_left_facet),
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
