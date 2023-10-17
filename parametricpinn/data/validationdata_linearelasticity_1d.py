from typing import TypeAlias

from parametricpinn.data.dataset.stretchedrod_1d import (
    StretchedRodValidationDataset1D,
    StretchedRodValidationDataset1DConfig,
    calculate_displacements_solution,
)
from parametricpinn.data.geometry import StretchedRod1D
from parametricpinn.errors import DatasetConfigError

ValidationDatasetConfig: TypeAlias = StretchedRodValidationDataset1DConfig
ValidationDataset: TypeAlias = StretchedRodValidationDataset1D


def create_validation_dataset(config: ValidationDatasetConfig) -> ValidationDataset:
    if isinstance(config, StretchedRodValidationDataset1DConfig):
        geometry = StretchedRod1D(length=config.length)
        return StretchedRodValidationDataset1D(
            geometry=geometry,
            traction=config.traction,
            volume_force=config.volume_force,
            min_youngs_modulus=config.min_youngs_modulus,
            max_youngs_modulus=config.max_youngs_modulus,
            num_points=config.num_points,
            num_samples=config.num_samples,
        )

    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
