import os
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest
import torch

from parametricpinn.data.validationdata_2d import (
    ValidationDataset2D,
    ValidationDataset2DConfig,
    create_validation_dataset,
)
from parametricpinn.errors import TestConfigurationError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import PandasDataWriter
from parametricpinn.settings import Settings, set_seed
from parametricpinn.types import Tensor


class FakeSetting(Settings):
    def __init__(self) -> None:
        self.PROJECT_DIR = Path() / "parametricpinn" / "data" / "tests"
        self.OUTPUT_SUBDIR = "test_output"
        self.INPUT_SUBDIR = "test_input"


random_seed = 0
settings = FakeSetting()
project_directory = ProjectDirectory(settings)
data_writer = PandasDataWriter(project_directory)
input_subdir = "test_input_subdirectory"
size_validation_data = 4
num_points = 3
num_samples = 2

coordinates_x = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape((-1, 1))
coordinates_y = torch.tensor([1.1, 2.1, 3.1, 4.1]).reshape((-1, 1))
youngs_modulus_0 = torch.tensor([11.0])
poissons_ratio_0 = torch.tensor([11.1])
youngs_modulus_1 = torch.tensor([21.0])
poissons_ratio_1 = torch.tensor([21.1])
displacements_x_0 = torch.tensor([31.0, 32.0, 33.0, 34.0]).reshape((-1, 1))
displacements_y_0 = torch.tensor([31.1, 32.1, 33.1, 34.1]).reshape((-1, 1))
displacements_x_1 = torch.tensor([41.0, 42.0, 43.0, 44.0]).reshape((-1, 1))
displacements_y_1 = torch.tensor([41.1, 42.1, 43.1, 44.1]).reshape((-1, 1))

coordinates_all = torch.concat((coordinates_x, coordinates_y), dim=1)
parameters_0_all = torch.concat((youngs_modulus_0, poissons_ratio_0)).repeat(
    (size_validation_data, 1)
)
parameters_1_all = torch.concat((youngs_modulus_1, poissons_ratio_1)).repeat(
    (size_validation_data, 1)
)
displacements_0_all = torch.concat((displacements_x_0, displacements_y_0), dim=1)
displacements_1_all = torch.concat((displacements_x_1, displacements_y_1), dim=1)


def write_input_data() -> None:
    displacement_input_0 = {
        "coordinates_x": np.ravel(coordinates_x.numpy()),
        "coordinates_y": np.ravel(coordinates_y.numpy()),
        "displacements_x": np.ravel(displacements_x_0.numpy()),
        "displacements_y": np.ravel(displacements_y_0.numpy()),
    }
    parameters_input_0 = {
        "youngs_modulus": youngs_modulus_0.numpy(),
        "poissons_ratio": poissons_ratio_0.numpy(),
    }
    output_subdir_0 = os.path.join(input_subdir, "sample_0")
    data_writer.write(
        pd.DataFrame(displacement_input_0),
        "displacements",
        output_subdir_0,
        header=True,
        save_to_input_dir=True,
    )
    data_writer.write(
        pd.DataFrame(parameters_input_0),
        "parameters",
        output_subdir_0,
        header=True,
        save_to_input_dir=True,
    )
    displacement_input_1 = {
        "coordinates_x": np.ravel(coordinates_x.numpy()),
        "coordinates_y": np.ravel(coordinates_y.numpy()),
        "displacements_x": np.ravel(displacements_x_1.numpy()),
        "displacements_y": np.ravel(displacements_y_1.numpy()),
    }
    parameters_input_1 = {
        "youngs_modulus": youngs_modulus_1.numpy(),
        "poissons_ratio": poissons_ratio_1.numpy(),
    }
    output_subdir_1 = os.path.join(input_subdir, "sample_1")
    data_writer.write(
        pd.DataFrame(displacement_input_1),
        "displacements",
        output_subdir_1,
        header=True,
        save_to_input_dir=True,
    )
    data_writer.write(
        pd.DataFrame(parameters_input_1),
        "parameters",
        output_subdir_1,
        header=True,
        save_to_input_dir=True,
    )


### Test ValidationDataset
@pytest.fixture
def sut() -> Iterator[ValidationDataset2D]:
    set_seed(random_seed)
    input_subdir_path = settings.PROJECT_DIR / settings.INPUT_SUBDIR
    if not Path.is_dir(input_subdir_path):
        os.mkdir(input_subdir_path)
    write_input_data()
    config = ValidationDataset2DConfig(
        input_subdir=input_subdir,
        num_points=num_points,
        num_samples=num_samples,
        project_directory=project_directory,
    )
    yield create_validation_dataset(config=config)
    shutil.rmtree(input_subdir_path)


def expected_sample(idx_sample: int) -> tuple[Tensor, Tensor]:
    # The random indices must be generated in the same order as in the system under test.
    set_seed(random_seed)
    # Sample 0
    random_indices_0 = torch.randperm(size_validation_data)[:num_points]
    coordinates_0 = coordinates_all[random_indices_0]
    parameters_0 = parameters_0_all[random_indices_0]
    displacements_0 = displacements_0_all[random_indices_0]
    sample_x_0 = torch.concat((coordinates_0, parameters_0), dim=1)
    sample_y_true_0 = displacements_0
    # Sample 1
    random_indices_1 = torch.randperm(size_validation_data)[:num_points]
    coordinates_1 = coordinates_all[random_indices_1]
    parameters_1 = parameters_1_all[random_indices_1]
    displacements_1 = displacements_1_all[random_indices_1]
    sample_x_1 = torch.concat((coordinates_1, parameters_1), dim=1)
    sample_y_true_1 = displacements_1
    # Return sample
    if idx_sample == 0:
        return sample_x_0, sample_y_true_0
    elif idx_sample == 1:
        return sample_x_1, sample_y_true_1
    else:
        raise TestConfigurationError(f"Sample index {idx_sample} not specified.")


def test_len(sut: ValidationDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_input_sample(
    sut: ValidationDataset2D,
    idx_sample: int,
) -> None:
    actual, _ = sut[idx_sample]

    expected, _ = expected_sample(idx_sample)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_output_sample(
    sut: ValidationDataset2D,
    idx_sample: int,
) -> None:
    _, actual = sut[idx_sample]

    _, expected = expected_sample(idx_sample)
    torch.testing.assert_close(actual, expected)


## Test collate_func()
@pytest.fixture
def fake_batch() -> list[tuple[Tensor, Tensor]]:
    sample_x_0 = torch.concat((coordinates_all, parameters_0_all), dim=1)
    sample_y_true_0 = displacements_0_all
    sample_x_1 = torch.concat((coordinates_all, parameters_1_all), dim=1)
    sample_y_true_1 = displacements_1_all
    return [(sample_x_0, sample_y_true_0), (sample_x_1, sample_y_true_1)]


def test_batch_pde__x(
    sut: ValidationDataset2D,
    fake_batch: list[tuple[Tensor, Tensor]],
):
    collate_func = sut.get_collate_func()

    actual, _ = collate_func(fake_batch)

    sample_0 = torch.concat((coordinates_all, parameters_0_all), dim=1)
    sample_1 = torch.concat((coordinates_all, parameters_1_all), dim=1)
    expected = torch.concat((sample_0, sample_1), dim=0)
    torch.testing.assert_close(actual, expected)


def test_batch_pde__y_true(
    sut: ValidationDataset2D,
    fake_batch: list[tuple[Tensor, Tensor]],
):
    collate_func = sut.get_collate_func()

    _, actual = collate_func(fake_batch)

    sample_0 = displacements_0_all
    sample_1 = displacements_1_all
    expected = torch.concat((sample_0, sample_1), dim=0)
    torch.testing.assert_close(actual, expected)
