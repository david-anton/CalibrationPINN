import os
import shutil
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest
import torch

from parametricpinn.data.dataset import SimulationData, SimulationDataList
from parametricpinn.data.simulation_2d import (
    SimulationDataset2D,
    SimulationDataset2DConfig,
    create_simulation_dataset,
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
parameter_0_sample_0 = torch.tensor([11.0])
parameter_1_sample_0 = torch.tensor([11.1])
parameter_0_sample_1 = torch.tensor([21.0])
parameter_1_sample_1 = torch.tensor([21.1])
displacements_x_0 = torch.tensor([31.0, 32.0, 33.0, 34.0]).reshape((-1, 1))
displacements_y_0 = torch.tensor([31.1, 32.1, 33.1, 34.1]).reshape((-1, 1))
displacements_x_1 = torch.tensor([41.0, 42.0, 43.0, 44.0]).reshape((-1, 1))
displacements_y_1 = torch.tensor([41.1, 42.1, 43.1, 44.1]).reshape((-1, 1))

coordinates_all = torch.concat((coordinates_x, coordinates_y), dim=1)
parameters_0_all = torch.concat((parameter_0_sample_0, parameter_1_sample_0)).repeat(
    (size_validation_data, 1)
)
parameters_1_all = torch.concat((parameter_0_sample_1, parameter_1_sample_1)).repeat(
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
        "parameter_0": parameter_0_sample_0.numpy(),
        "parameter_1": parameter_1_sample_0.numpy(),
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
        "parameter_0": parameter_0_sample_1.numpy(),
        "parameetr_1": parameter_1_sample_1.numpy(),
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


### Test SimulationDataset
@pytest.fixture
def sut() -> Iterator[SimulationDataset2D]:
    set_seed(random_seed)
    input_subdir_path = settings.PROJECT_DIR / settings.INPUT_SUBDIR
    if not Path.is_dir(input_subdir_path):
        os.mkdir(input_subdir_path)
    write_input_data()
    config = SimulationDataset2DConfig(
        input_subdir=input_subdir,
        num_points=num_points,
        num_samples=num_samples,
        project_directory=project_directory,
    )
    yield create_simulation_dataset(config=config)
    shutil.rmtree(input_subdir_path)


def generate_expected_sample(idx_sample: int) -> SimulationData:
    # The random indices must be generated in the same order as in the system under test.
    set_seed(random_seed)
    # Sample 0
    random_indices_0 = torch.randperm(size_validation_data)[:num_points]
    x_coor_0 = coordinates_all[random_indices_0]
    x_params_0 = parameters_0_all[random_indices_0]
    y_true_0 = displacements_0_all[random_indices_0]
    sample_0 = SimulationData(x_coor=x_coor_0, x_params=x_params_0, y_true=y_true_0)
    # Sample 1
    random_indices_1 = torch.randperm(size_validation_data)[:num_points]
    x_coor_1 = coordinates_all[random_indices_1]
    x_params_1 = parameters_1_all[random_indices_1]
    y_true_1 = displacements_1_all[random_indices_1]
    sample_1 = SimulationData(x_coor=x_coor_1, x_params=x_params_1, y_true=y_true_1)
    # Return sample
    if idx_sample == 0:
        return sample_0
    elif idx_sample == 1:
        return sample_1
    else:
        raise TestConfigurationError(f"Sample index {idx_sample} not specified.")


def test_len(sut: SimulationDataset2D) -> None:
    actual = len(sut)

    expected = num_samples
    assert actual == expected


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_x_coor(
    sut: SimulationDataset2D,
    idx_sample: int,
) -> None:
    sample = sut[idx_sample]
    actual = sample.x_coor

    expected_sample = generate_expected_sample(idx_sample)
    expected = expected_sample.x_coor
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_x_params(
    sut: SimulationDataset2D,
    idx_sample: int,
) -> None:
    sample = sut[idx_sample]
    actual = sample.x_params

    expected_sample = generate_expected_sample(idx_sample)
    expected = expected_sample.x_params
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(("idx_sample"), range(num_samples))
def test_y_true(
    sut: SimulationDataset2D,
    idx_sample: int,
) -> None:
    sample = sut[idx_sample]
    actual = sample.y_true

    expected_sample = generate_expected_sample(idx_sample)
    expected = expected_sample.y_true
    torch.testing.assert_close(actual, expected)


## Test collate_func()
@pytest.fixture
def fake_batch() -> SimulationDataList:
    sample_0 = SimulationData(
        x_coor=coordinates_all, x_params=parameters_0_all, y_true=displacements_0_all
    )
    sample_1 = SimulationData(
        x_coor=coordinates_all, x_params=parameters_1_all, y_true=displacements_1_all
    )
    return [sample_0, sample_1]


def test_batch__x_coor(
    sut: SimulationDataset2D,
    fake_batch: SimulationDataList,
):
    collate_func = sut.get_collate_func()

    batched_data = collate_func(fake_batch)
    actual = batched_data.x_coor

    expected = torch.concat((coordinates_all, coordinates_all), dim=0)
    torch.testing.assert_close(actual, expected)


def test_batch__x_params(
    sut: SimulationDataset2D,
    fake_batch: SimulationDataList,
):
    collate_func = sut.get_collate_func()

    batched_data = collate_func(fake_batch)
    actual = batched_data.x_params

    expected = torch.concat((parameters_0_all, parameters_1_all), dim=0)
    torch.testing.assert_close(actual, expected)


def test_batch__y_true(
    sut: SimulationDataset2D,
    fake_batch: SimulationDataList,
):
    collate_func = sut.get_collate_func()

    batched_data = collate_func(fake_batch)
    actual = batched_data.y_true

    expected = torch.concat((displacements_0_all, displacements_1_all), dim=0)
    torch.testing.assert_close(actual, expected)
