from typing import TypeAlias, Union

from parametricpinn.data.dataset import (
    DogBoneTrainingDataset2D,
    DogBoneTrainingDataset2DConfig,
    PlateWithHoleTrainingDataset2D,
    PlateWithHoleTrainingDataset2DConfig,
    QuarterPlateWithHoleTrainingDataset2D,
    QuarterPlateWithHoleTrainingDataset2DConfig,
)
from parametricpinn.data.geometry import (
    DogBone2D,
    DogBoneGeometryConfig,
    PlateWithHole2D,
    QuarterPlateWithHole2D,
)
from parametricpinn.errors import DatasetConfigError

TrainingDatasetConfig: TypeAlias = Union[
    QuarterPlateWithHoleTrainingDataset2DConfig,
    PlateWithHoleTrainingDataset2DConfig,
    DogBoneTrainingDataset2DConfig,
]
TrainingDataset: TypeAlias = Union[
    QuarterPlateWithHoleTrainingDataset2D,
    PlateWithHoleTrainingDataset2D,
    DogBoneTrainingDataset2D,
]


def create_training_dataset(config: TrainingDatasetConfig) -> TrainingDataset:
    if isinstance(config, QuarterPlateWithHoleTrainingDataset2DConfig):
        geometry_quarter_pwh = QuarterPlateWithHole2D(
            edge_length=config.edge_length, radius=config.radius
        )
        return QuarterPlateWithHoleTrainingDataset2D(
            geometry=geometry_quarter_pwh,
            traction_left=config.traction_left,
            volume_force=config.volume_force,
            min_youngs_modulus=config.min_youngs_modulus,
            max_youngs_modulus=config.max_youngs_modulus,
            min_poissons_ratio=config.min_poissons_ratio,
            max_poissons_ratio=config.max_poissons_ratio,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            num_samples_per_parameter=config.num_samples_per_parameter,
        )
    elif isinstance(config, PlateWithHoleTrainingDataset2DConfig):
        geometry_pwh = PlateWithHole2D(
            plate_length=config.plate_length,
            plate_height=config.plate_height,
            hole_radius=config.hole_radius,
        )
        return PlateWithHoleTrainingDataset2D(
            geometry=geometry_pwh,
            traction_right=config.traction_right,
            volume_force=config.volume_force,
            min_youngs_modulus=config.min_youngs_modulus,
            max_youngs_modulus=config.max_youngs_modulus,
            min_poissons_ratio=config.min_poissons_ratio,
            max_poissons_ratio=config.max_poissons_ratio,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            num_samples_per_parameter=config.num_samples_per_parameter,
        )
    elif isinstance(config, DogBoneTrainingDataset2DConfig):
        geometry_config_dogbone = DogBoneGeometryConfig()
        geometry_dogbone = DogBone2D(geometry_config_dogbone)
        return DogBoneTrainingDataset2D(
            geometry=geometry_dogbone,
            traction_right=config.traction_right,
            volume_force=config.volume_force,
            min_youngs_modulus=config.min_youngs_modulus,
            max_youngs_modulus=config.max_youngs_modulus,
            min_poissons_ratio=config.min_poissons_ratio,
            max_poissons_ratio=config.max_poissons_ratio,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            num_samples_per_parameter=config.num_samples_per_parameter,
        )
    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
