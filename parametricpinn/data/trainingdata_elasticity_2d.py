from typing import TypeAlias

from parametricpinn.data.dataset.dataset import TrainingDataset
from parametricpinn.data.dataset.quarterplatewithholedatasets import (
    QuarterPlateWithHoleTrainingDataset,
    QuarterPlateWithHoleTrainingDatasetConfig,
)
from parametricpinn.data.geometry import PlateWithHole, QuarterPlateWithHole
from parametricpinn.errors import DatasetConfigError

TrainingDatasetConfig: TypeAlias = QuarterPlateWithHoleTrainingDatasetConfig


def create_training_dataset_2D(
    training_dataset_config: TrainingDatasetConfig,
) -> TrainingDataset:
    config = training_dataset_config
    if isinstance(config, QuarterPlateWithHoleTrainingDatasetConfig):
        geometry = QuarterPlateWithHole(
            edge_length=config.edge_length, radius=config.radius
        )
        return QuarterPlateWithHoleTrainingDataset(
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
