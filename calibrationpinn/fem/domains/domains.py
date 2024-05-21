from typing import Protocol, TypeAlias, Union

from calibrationpinn.errors import FEMDomainConfigError
from calibrationpinn.fem.base import (
    DFunctionSpace,
    DMesh,
    DMeshTags,
    UFLMeasure,
    UFLTestFunction,
)
from calibrationpinn.fem.boundaryconditions import BoundaryConditions
from calibrationpinn.fem.domains.dogbone import DogBoneDomain, DogBoneDomainConfig
from calibrationpinn.fem.domains.plate import PlateDomain, PlateDomainConfig
from calibrationpinn.fem.domains.platewithhole import (
    PlateWithHoleDomain,
    PlateWithHoleDomainConfig,
)
from calibrationpinn.fem.domains.quarterplatewithhole import (
    QuarterPlateWithHoleDomain,
    QuarterPlateWithHoleDomainConfig,
)
from calibrationpinn.fem.domains.simplifieddogbone import (
    SimplifiedDogBoneDomain,
    SimplifiedDogBoneDomainConfig,
)
from calibrationpinn.io import ProjectDirectory

DomainConfig: TypeAlias = Union[
    QuarterPlateWithHoleDomainConfig,
    PlateWithHoleDomainConfig,
    PlateDomainConfig,
    DogBoneDomainConfig,
    SimplifiedDogBoneDomainConfig,
]


class Domain(Protocol):
    config: DomainConfig
    mesh: DMesh
    boundary_tags: DMeshTags

    def define_boundary_conditions(
        self,
        function_space: DFunctionSpace,
        measure: UFLMeasure,
        test_function: UFLTestFunction,
    ) -> BoundaryConditions:
        pass


def create_domain(
    domain_config: DomainConfig,
    save_mesh: bool,
    output_subdir: str,
    project_directory: ProjectDirectory,
    save_to_input_dir: bool = False,
) -> Domain:
    if isinstance(domain_config, QuarterPlateWithHoleDomainConfig):
        return QuarterPlateWithHoleDomain(
            config=domain_config,
            save_mesh=save_mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
    elif isinstance(domain_config, PlateWithHoleDomainConfig):
        return PlateWithHoleDomain(
            config=domain_config,
            save_mesh=save_mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
    elif isinstance(domain_config, PlateDomainConfig):
        return PlateDomain(
            config=domain_config,
            save_mesh=save_mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
    elif isinstance(domain_config, DogBoneDomainConfig):
        return DogBoneDomain(
            config=domain_config,
            save_mesh=save_mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
    elif isinstance(domain_config, SimplifiedDogBoneDomainConfig):
        return SimplifiedDogBoneDomain(
            config=domain_config,
            save_mesh=save_mesh,
            output_subdir=output_subdir,
            project_directory=project_directory,
            save_to_input_dir=save_to_input_dir,
        )
    else:
        raise FEMDomainConfigError(
            f"There is no implementation for the requested FEM domain {domain_config}."
        )
