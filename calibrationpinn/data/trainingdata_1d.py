from typing import TypeAlias

from calibrationpinn.data.dataset import (
    StretchedRodTrainingDataset1D,
    StretchedRodTrainingDataset1DConfig,
)
from calibrationpinn.data.geometry import StretchedRod1D
from calibrationpinn.errors import DatasetConfigError

TrainingDatasetConfig: TypeAlias = StretchedRodTrainingDataset1DConfig
TrainingDataset: TypeAlias = StretchedRodTrainingDataset1D


def create_training_dataset(config: TrainingDatasetConfig) -> TrainingDataset:
    if isinstance(config, StretchedRodTrainingDataset1DConfig):
        geometry = StretchedRod1D(length=config.length)
        return StretchedRodTrainingDataset1D(
            parameters_samples=config.parameters_samples,
            geometry=geometry,
            traction=config.traction,
            volume_force=config.volume_force,
            num_points_pde=config.num_points_pde,
        )
    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
