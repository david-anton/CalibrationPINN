from typing import TypeAlias

from parametricpinn.data.dataset import (
    StretchedRodSimulationDatasetLinearElasticity1D,
    StretchedRodSimulationDatasetLinearElasticity1DConfig,
)
from parametricpinn.data.dataset.simulationdataset_stretchedrod_1d import (
    LinearElasticDispalcementSolutionFunc,
    calculate_linear_elastic_displacements_solution,
)
from parametricpinn.data.geometry import StretchedRod1D
from parametricpinn.errors import DatasetConfigError

ValidationDatasetConfig: TypeAlias = (
    StretchedRodSimulationDatasetLinearElasticity1DConfig
)
ValidationDataset: TypeAlias = StretchedRodSimulationDatasetLinearElasticity1D


def create_simulation_dataset(config: ValidationDatasetConfig) -> ValidationDataset:
    if isinstance(config, StretchedRodSimulationDatasetLinearElasticity1DConfig):
        geometry = StretchedRod1D(length=config.length)
        return StretchedRodSimulationDatasetLinearElasticity1D(
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
