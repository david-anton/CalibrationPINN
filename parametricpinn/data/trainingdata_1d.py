from typing import TypeAlias

from parametricpinn.data.dataset import (
    StretchedRodTrainingDataset1D,
    StretchedRodTrainingDataset1DConfig,
)
from parametricpinn.data.geometry import StretchedRod1D
from parametricpinn.errors import DatasetConfigError

TrainingDatasetConfig: TypeAlias = StretchedRodTrainingDataset1DConfig
TrainingDataset: TypeAlias = StretchedRodTrainingDataset1D


def create_training_dataset(config: TrainingDatasetConfig) -> TrainingDataset:
    if isinstance(config, StretchedRodTrainingDataset1DConfig):
        geometry = StretchedRod1D(length=config.length)
        return StretchedRodTrainingDataset1D(
            parameters=config.parameters,
            geometry=geometry,
            traction=config.traction,
            volume_force=config.volume_force,
            num_points_pde=config.num_points_pde,
            num_samples=config.num_samples,
        )
    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
