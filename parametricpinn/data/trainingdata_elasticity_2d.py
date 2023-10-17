from typing import TypeAlias

from parametricpinn.data.dataset import (
    QuarterPlateWithHoleTrainingDataset2D,
    QuarterPlateWithHoleTrainingDataset2DConfig,
)
from parametricpinn.data.geometry import PlateWithHole2D, QuarterPlateWithHole2D
from parametricpinn.errors import DatasetConfigError

TrainingDatasetConfig: TypeAlias = QuarterPlateWithHoleTrainingDataset2DConfig
TrainingDataset: TypeAlias = QuarterPlateWithHoleTrainingDataset2D


def create_training_dataset(config: TrainingDatasetConfig) -> TrainingDataset:
    if isinstance(config, QuarterPlateWithHoleTrainingDataset2DConfig):
        geometry = QuarterPlateWithHole2D(
            edge_length=config.edge_length, radius=config.radius
        )
        return QuarterPlateWithHoleTrainingDataset2D(
            geometry=geometry,
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
    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
