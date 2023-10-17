from typing import TypeAlias

from parametricpinn.data.dataset.quarterplatewithholedatasets_2d import (
    QuarterPlateWithHoleValidationDataset2D,
    QuarterPlateWithHoleValidationDataset2DConfig,
)
from parametricpinn.errors import DatasetConfigError

ValidationDatasetConfig: TypeAlias = QuarterPlateWithHoleValidationDataset2DConfig
ValidationDataset: TypeAlias = QuarterPlateWithHoleValidationDataset2D


def create_validation_dataset(config: ValidationDatasetConfig) -> ValidationDataset:
    if isinstance(config, QuarterPlateWithHoleValidationDataset2DConfig):
        return QuarterPlateWithHoleValidationDataset2D(
            input_subdir=config.input_subdir,
            num_points=config.num_points,
            num_samples=config.num_samples,
            project_directory=config.project_directory,
        )

    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
