import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.dataset import (
    SimulationCollateFunc,
    SimulationData,
    SimulationDataList,
)
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.types import Tensor


@dataclass
class SimulationDataset2DConfig:
    input_subdir: str
    num_points: int
    num_samples: int
    project_directory: ProjectDirectory
    read_from_output_dir: bool = False


class SimulationDataset2D(Dataset):
    def __init__(
        self,
        input_subdir: str,
        num_points: int,
        num_samples: int,
        project_directory: ProjectDirectory,
        read_from_output_dir: bool = False,
    ) -> None:
        self._input_subdir = input_subdir
        self._num_points = num_points
        self._num_samples = num_samples
        self._data_reader = CSVDataReader(project_directory)
        self._read_from_output_dir = read_from_output_dir
        self._file_name_displacements = "displacements"
        self._file_name_parameters = "parameters"
        self._slice_coordinates = slice(0, 2)
        self._slice_displacements = slice(2, 4)
        self._samples: SimulationDataList = []

        self._load_samples()

    def get_collate_func(self) -> SimulationCollateFunc:
        def collate_func(batch: SimulationDataList) -> SimulationData:
            x_coor_batch = []
            x_params_batch = []
            y_true_batch = []

            def append_to_batch(sample: SimulationData) -> None:
                x_coor_batch.append(sample.x_coor)
                x_params_batch.append(sample.x_params)
                y_true_batch.append(sample.y_true)

            for sample in batch:
                append_to_batch(sample)

            batched_data = SimulationData(
                x_coor=torch.concat(x_coor_batch, dim=0),
                x_params=torch.concat(x_params_batch, dim=0),
                y_true=torch.concat(y_true_batch, dim=0),
            )
            return batched_data

        return collate_func

    def _load_samples(self) -> None:
        for idx_sample in range(self._num_samples):
            displacements = self._read_data(self._file_name_displacements, idx_sample)
            parameters = self._read_data(self._file_name_parameters, idx_sample)
            random_indices = self._generate_random_indices(displacements.size(dim=0))
            self._add_sample(displacements, parameters, random_indices)

    def _read_data(self, file_name, idx_sample) -> Tensor:
        sample_subdir = f"sample_{idx_sample}"
        input_subdir = os.path.join(self._input_subdir, sample_subdir)
        data = self._data_reader.read(
            file_name, input_subdir, read_from_output_dir=self._read_from_output_dir
        )
        return torch.tensor(data, dtype=torch.get_default_dtype())

    def _generate_random_indices(self, max_index: int):
        return torch.randperm(max_index)[: self._num_points]

    def _add_sample(
        self, displacements: Tensor, parameters: Tensor, random_indices: Tensor
    ) -> None:
        x_coor_all = displacements[:, self._slice_coordinates]
        x_coor = x_coor_all[random_indices, :]
        x_params = repeat_tensor(parameters, (self._num_points, 1))
        y_true_all = displacements[:, self._slice_displacements]
        y_true = y_true_all[random_indices, :]
        sample = SimulationData(x_coor=x_coor, x_params=x_params, y_true=y_true)
        self._samples.append(sample)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> SimulationData:
        return self._samples[idx]
