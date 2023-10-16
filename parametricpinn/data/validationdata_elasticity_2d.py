from typing import TypeAlias

from parametricpinn.data.dataset.dataset import ValidationDataset
from parametricpinn.data.dataset.quarterplatewithholedatasets import (
    QuarterPlateWithHoleValidationDataset,
    QuarterPlateWithHoleValidationDatasetConfig,
)
from parametricpinn.errors import DatasetConfigError

ValidationDatasetConfig: TypeAlias = QuarterPlateWithHoleValidationDatasetConfig


def create_validation_dataset_2D(
    validation_dataset_config: ValidationDatasetConfig,
) -> ValidationDataset:
    config = validation_dataset_config
    if isinstance(config, QuarterPlateWithHoleValidationDatasetConfig):
        return QuarterPlateWithHoleValidationDataset(
            input_subdir=config.input_subdir,
            num_points=config.num_points,
            num_samples=config.num_samples,
            project_directory=config.project_directory,
        )

    else:
        raise DatasetConfigError(
            f"There is no implementation for the requested dataset configuration {config}."
        )
