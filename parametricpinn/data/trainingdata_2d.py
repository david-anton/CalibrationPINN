from typing import TypeAlias, Union

from parametricpinn.data.dataset import (
    DogBoneTrainingDataset2D,
    DogBoneTrainingDataset2DConfig,
    PlateTrainingDataset2D,
    PlateTrainingDataset2DConfig,
    PlateWithHoleTrainingDataset2D,
    PlateWithHoleTrainingDataset2DConfig,
    QuarterPlateWithHoleTrainingDataset2D,
    QuarterPlateWithHoleTrainingDataset2DConfig,
    SimplifiedDogBoneTrainingDataset2D,
    SimplifiedDogBoneTrainingDataset2DConfig,
)
from parametricpinn.data.geometry import (
    DogBone2D,
    DogBoneGeometryConfig,
    Plate2D,
    PlateWithHole2D,
    QuarterPlateWithHole2D,
    SimplifiedDogBone2D,
    SimplifiedDogBoneGeometryConfig,
)
from parametricpinn.errors import DatasetConfigError

TrainingDatasetConfig: TypeAlias = Union[
    QuarterPlateWithHoleTrainingDataset2DConfig,
    PlateWithHoleTrainingDataset2DConfig,
    PlateTrainingDataset2DConfig,
    DogBoneTrainingDataset2DConfig,
    SimplifiedDogBoneTrainingDataset2DConfig,
]
TrainingDataset: TypeAlias = Union[
    QuarterPlateWithHoleTrainingDataset2D,
    PlateWithHoleTrainingDataset2D,
    PlateTrainingDataset2D,
    DogBoneTrainingDataset2D,
    SimplifiedDogBoneTrainingDataset2D,
]


def create_training_dataset(config: TrainingDatasetConfig) -> TrainingDataset:
    if isinstance(config, QuarterPlateWithHoleTrainingDataset2DConfig):
        geometry_quarter_pwh = QuarterPlateWithHole2D(
            edge_length=config.edge_length, radius=config.radius
        )
        return QuarterPlateWithHoleTrainingDataset2D(
            parameters_samples=config.parameters_samples,
            geometry=geometry_quarter_pwh,
            traction_left=config.traction_left,
            volume_force=config.volume_force,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            bcs_overlap_distance=config.bcs_overlap_distance,
            bcs_overlap_angle_distance=config.bcs_overlap_angle_distance,
        )
    elif isinstance(config, PlateWithHoleTrainingDataset2DConfig):
        geometry_pwh = PlateWithHole2D(
            plate_length=config.plate_length,
            plate_height=config.plate_height,
            hole_radius=config.hole_radius,
        )
        return PlateWithHoleTrainingDataset2D(
            parameters_samples=config.parameters_samples,
            geometry=geometry_pwh,
            traction_right=config.traction_right,
            volume_force=config.volume_force,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            bcs_overlap_distance=config.bcs_overlap_distance,
        )
    elif isinstance(config, PlateTrainingDataset2DConfig):
        geometry = Plate2D(
            plate_length=config.plate_length,
            plate_height=config.plate_height,
        )
        return PlateTrainingDataset2D(
            parameters_samples=config.parameters_samples,
            geometry=geometry,
            traction_right=config.traction_right,
            volume_force=config.volume_force,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            bcs_overlap_distance=config.bcs_overlap_distance,
        )
    elif isinstance(config, DogBoneTrainingDataset2DConfig):
        geometry_config_dogbone = DogBoneGeometryConfig()
        geometry_dogbone = DogBone2D(geometry_config_dogbone)
        return DogBoneTrainingDataset2D(
            parameters_samples=config.parameters_samples,
            geometry=geometry_dogbone,
            traction_right=config.traction_right,
            volume_force=config.volume_force,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            bcs_overlap_angle_distance=config.bcs_overlap_angle_distance,
        )
    elif isinstance(config, SimplifiedDogBoneTrainingDataset2DConfig):
        geometry_config_simplified_dogbone = SimplifiedDogBoneGeometryConfig()
        geometry_simplified_dogbone = SimplifiedDogBone2D(
            geometry_config_simplified_dogbone
        )
        return SimplifiedDogBoneTrainingDataset2D(
            parameters_samples=config.parameters_samples,
            geometry=geometry_simplified_dogbone,
            traction_right=config.traction_right,
            volume_force=config.volume_force,
            num_collocation_points=config.num_collocation_points,
            num_points_per_bc=config.num_points_per_bc,
            bcs_overlap_angle_distance_left=config.bcs_overlap_angle_distance_left,
            bcs_overlap_distance_parallel_right=config.bcs_overlap_distance_parallel_right,
        )
    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
