from typing import Protocol, TypeAlias

from parametricpinn.errors import GeometryConfigError
from parametricpinn.fem.base import DMesh
from parametricpinn.fem.geometries.platewithhole import (
    PlateWithHoleGeometry,
    PlateWithHoleGeometryConfig,
)
from parametricpinn.io import ProjectDirectory

Config: TypeAlias = PlateWithHoleGeometryConfig


class Geometry(Protocol):
    def generate_mesh(
            self, 
            save_mesh: bool, 
            output_subdir: str, 
            project_directory: ProjectDirectory, 
            save_to_input_dir: bool = False
        ) -> DMesh:
        pass



def create_geometry(config: Config):
    if isinstance(config, PlateWithHoleGeometryConfig):
        return PlateWithHoleGeometry(config)
    
    else:
        raise GeometryConfigError(
            f"There is no implementation for the requested FEM geometry {config}."
        )