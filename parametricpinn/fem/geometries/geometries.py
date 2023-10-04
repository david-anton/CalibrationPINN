from typing import Protocol, TypeAlias

from parametricpinn.errors import FEMGeometryConfigError
from parametricpinn.fem.base import (
    DFunctionSpace,
    DMesh,
    DMeshTags,
    UFLMeasure,
    UFLTestFunction,
)
from parametricpinn.fem.boundaryconditions import BoundaryConditions
from parametricpinn.fem.geometries.platewithhole import (
    PlateWithHoleGeometry,
    PlateWithHoleGeometryConfig,
)
from parametricpinn.io import ProjectDirectory

GeometryConfig: TypeAlias = PlateWithHoleGeometryConfig


class Geometry(Protocol):
    def generate_mesh(
        self,
        save_mesh: bool,
        output_subdir: str,
        project_directory: ProjectDirectory,
        save_to_input_dir: bool = False,
    ) -> DMesh:
        pass

    def tag_boundaries(self, mesh: DMesh) -> DMeshTags:
        pass

    def define_boundary_conditions(
        self,
        mesh: DMesh,
        boundary_tags: DMeshTags,
        function_space: DFunctionSpace,
        measure: UFLMeasure,
        test_function: UFLTestFunction,
    ) -> BoundaryConditions:
        pass


def create_geometry(geometry_config: GeometryConfig):
    if isinstance(geometry_config, PlateWithHoleGeometryConfig):
        return PlateWithHoleGeometry(geometry_config)

    else:
        raise FEMGeometryConfigError(
            f"There is no implementation for the requested FEM geometry {geometry_config}."
        )
