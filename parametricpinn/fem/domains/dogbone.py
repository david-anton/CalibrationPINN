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
class DogBoneDomainConfigBase:
    origin_x = 0
    origin_y = 0
    box_length = 120
    box_height = 30
    half_box_length = box_length / 2
    half_box_height = box_height / 2
    parallel_length = 90
    parallel_height = 20
    half_parallel_length = parallel_length / 2
    half_parallel_height = parallel_height / 2
    cut_parallel_height = (box_height - parallel_height) / 2
    tapered_radius = 25
    plate_hole_radius = 4

@dataclass
class DogBoneDomainConfig(DogBoneDomainConfigBase):
    traction_right_x: float
    traction_right_y: float
    element_family: str
    element_degree: int
    element_size: float


class DogBoneDomain:
    def __init__(
        self,
        config: DogBoneDomainConfig,
        save_mesh: bool,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ):
        self.config = config
        self._geometric_dim = 2
        self._u_x_left = default_scalar_type(0.0)
        self._u_y_left = default_scalar_type(0.0)
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
        u_left = Constant(
            self.mesh, default_scalar_type(self._u_x_left, self._u_y_left)
        )
        traction_right = Constant(
            self.mesh,
            (
                default_scalar_type(
                    (self.config.traction_right_x, self.config.traction_right_y)
                )
            ),
        )
        return [
            DirichletBC(
                tag=self._tag_left,
                value=u_left,
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
        origin_x = self.config.origin_x 
        origin_y = self.config.origin_y
        box_length = self.config.box_length 
        box_height = self.config.box_height
        half_box_length = self.config.half_box_length 
        half_box_height = self.config.half_box_height 
        parallel_length = self.config.parallel_length 
        parallel_height = self.config.parallel_height
        half_parallel_length = self.config.half_parallel_length
        half_parallel_height = self.config.half_parallel_height 
        cut_parallel_height = self.config.cut_parallel_height 
        tapered_radius = self.config.tapered_radius 
        plate_hole_radius = self.config.plate_hole_radius 
        element_size = self.config.element_size
        geometry_kernel = gmsh.model.occ
        solid_marker = 1

        def create_geometry() -> GGeometry:
            gmsh.model.add("domain")
            box = geometry_kernel.add_rectangle(
                -half_box_length, -half_box_height, 0, box_length, box_height
            )
            cut_parallel_top = geometry_kernel.add_rectangle(
                -half_parallel_length,
                half_parallel_height,
                0,
                parallel_length,
                cut_parallel_height,
            )
            cut_parallel_bottom = geometry_kernel.add_rectangle(
                -half_parallel_length,
                -half_box_height,
                0,
                parallel_length,
                cut_parallel_height,
            )
            cut_tapered_top_left = geometry_kernel.add_disk(
                -half_parallel_length,
                half_parallel_height + tapered_radius,
                0,
                tapered_radius,
                tapered_radius,
            )
            cut_tapered_top_right = geometry_kernel.add_disk(
                half_parallel_length,
                half_parallel_height + tapered_radius,
                0,
                tapered_radius,
                tapered_radius,
            )
            cut_tapered_bottom_left = geometry_kernel.add_disk(
                -half_parallel_length,
                -half_parallel_height - tapered_radius,
                0,
                tapered_radius,
                tapered_radius,
            )
            cut_tapered_bottom_right = geometry_kernel.add_disk(
                half_parallel_length,
                -half_parallel_height - tapered_radius,
                0,
                tapered_radius,
                tapered_radius,
            )
            plate_hole = geometry_kernel.add_disk(
                origin_x, origin_y, 0, plate_hole_radius, plate_hole_radius
            )
            return geometry_kernel.cut(
                [(2, box)],
                [
                    (2, cut_parallel_top),
                    (2, cut_parallel_bottom),
                    (2, cut_tapered_top_left),
                    (2, cut_tapered_top_right),
                    (2, cut_tapered_bottom_left),
                    (2, cut_tapered_bottom_right),
                    (2, plate_hole),
                ],
            )

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
        locate_left_facet = lambda x: np.isclose(x[0], -60.0)
        locate_right_facet = lambda x: np.isclose(x[0], 60.0)
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
