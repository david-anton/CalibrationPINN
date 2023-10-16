import os

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.types import NPArray, Tensor


class QuarterPlateWithHoleValidationDataset2D(Dataset):
    def __init__(
        self,
        input_subdir: str,
        num_points: int,
        num_samples: int,
        project_directory: ProjectDirectory,
    ) -> None:
        self._input_subdir = input_subdir
        self._num_points = num_points
        self._num_samples = num_samples
        self._data_reader = CSVDataReader(project_directory)
        self._file_name_displacements = "displacements"
        self._file_name_parameters = "parameters"
        self._slice_coordinates = slice(0, 2)
        self._slice_displacements = slice(2, 4)
        self._samples_x: list[Tensor] = []
        self._samples_y_true: list[Tensor] = []

        self._load_samples()

    def _load_samples(self) -> None:
        for idx_sample in range(self._num_samples):
            displacements = self._read_data(self._file_name_displacements, idx_sample)
            parameters = self._read_data(self._file_name_parameters, idx_sample)
            random_indices = self._generate_random_indices(displacements.size(dim=0))
            self._add_input_sample(displacements, parameters, random_indices)
            self._add_output_sample(displacements, random_indices)

    def _read_data(self, file_name, idx_sample) -> Tensor:
        sample_subdir = f"sample_{idx_sample}"
        input_subdir = os.path.join(self._input_subdir, sample_subdir)
        data = self._data_reader.read(file_name, input_subdir)
        return torch.tensor(data, dtype=torch.get_default_dtype())

    def _generate_random_indices(self, max_index: int):
        return torch.randperm(max_index)[: self._num_points]

    def _add_input_sample(
        self, displacements: Tensor, parameters: Tensor, random_indices: Tensor
    ) -> None:
        x_coor_all = displacements[:, self._slice_coordinates]
        x_coor = x_coor_all[random_indices, :]
        x_params = repeat_tensor(parameters, (self._num_points, 1))
        x = torch.concat((x_coor, x_params), dim=1)
        self._samples_x.append(x)

    def _add_output_sample(self, displacements: Tensor, random_indices: Tensor) -> None:
        y_true_all = displacements[:, self._slice_displacements]
        y_true = y_true_all[random_indices, :]
        self._samples_y_true.append(y_true)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample_x = self._samples_x[idx]
        sample_y_true = self._samples_y_true[idx]
        return sample_x, sample_y_true


def collate_validation_data_2D(
    batch: list[tuple[Tensor, Tensor]]
) -> tuple[Tensor, Tensor]:
    x_batch = []
    y_true_batch = []

    for sample_x, sample_y_true in batch:
        x_batch.append(sample_x)
        y_true_batch.append(sample_y_true)

    batch_x = torch.concat(x_batch, dim=0)
    batch_y_true = torch.concat(y_true_batch, dim=0)
    return batch_x, batch_y_true


def create_validation_dataset_2D(
    input_subdir: str,
    num_points: int,
    num_samples: int,
    project_directory: ProjectDirectory,
):
    return QuarterPlateWithHoleValidationDataset2D(
        input_subdir=input_subdir,
        num_points=num_points,
        num_samples=num_samples,
        project_directory=project_directory,
    )
