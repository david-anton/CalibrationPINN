import os
from dataclasses import dataclass
from typing import TypeAlias

import torch

from parametricpinn.errors import UnvalidCalibrationDataError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.types import Tensor

Parameters: TypeAlias = Tensor


@dataclass
class CalibrationData:
    inputs: Tensor
    outputs: Tensor
    std_noise: float


class CalibrationDataLoader2D:
    def __init__(
        self,
        input_subdir: str,
        num_data_points: int,
        std_noise: float,
        num_cases: int,
        project_directory: ProjectDirectory,
    ) -> None:
        self._input_subdir = input_subdir
        self._num_data_points = num_data_points
        self._std_noise = std_noise
        self._num_cases = num_cases
        self._data_reader = CSVDataReader(project_directory)
        self._file_name_displacements = "displacements"
        self._file_name_parameters = "parameters"
        self._slice_coordinates = slice(0, 2)
        self._slice_displacements = slice(2, 4)
        self._data_sets: list[CalibrationData] = []
        self._true_parameters: list[Parameters] = []
        self._load_data()

    def get_data(self) -> tuple[tuple[CalibrationData, ...], Tensor]:
        data_sets = tuple(self._data_sets)
        true_parameters = torch.concat(self._true_parameters, dim=0)
        return data_sets, true_parameters

    def _load_data(self) -> None:
        for idx_case in range(self._num_cases):
            displacements = self._read_data(self._file_name_displacements, idx_case)
            true_parameters = self._read_data(self._file_name_parameters, idx_case)
            random_indices = self._generate_random_indices(displacements.size(dim=0))
            self._add_data_set(displacements, random_indices)
            self._add_true_parameters(true_parameters)

    def _read_data(self, file_name, idx_sample) -> Tensor:
        sample_subdir = f"sample_{idx_sample}"
        input_subdir = os.path.join(self._input_subdir, sample_subdir)
        data = self._data_reader.read(file_name, input_subdir)
        return torch.tensor(data, dtype=torch.get_default_dtype())

    def _generate_random_indices(self, max_index: int):
        return torch.randperm(max_index)[: self._num_data_points]

    def _add_data_set(self, displacements: Tensor, random_indices: Tensor) -> None:
        inputs_all = displacements[:, self._slice_coordinates]
        inputs = inputs_all[random_indices, :]
        outputs_all = displacements[:, self._slice_displacements]
        outputs = outputs_all[random_indices, :]
        noisy_outputs = outputs + torch.normal(
            mean=0.0, std=self._std_noise, size=outputs.size()
        )
        data_set = CalibrationData(
            inputs=inputs, outputs=noisy_outputs, std_noise=self._std_noise
        )
        self._data_sets.append(data_set)

    def _add_true_parameters(self, true_parameters: Tensor) -> None:
        self._true_parameters.append(true_parameters)


@dataclass
class PreprocessedCalibrationData(CalibrationData):
    num_data_points: int
    dim_outputs: int


def preprocess_calibration_data(data: CalibrationData) -> PreprocessedCalibrationData:
    _validate_calibration_data(data)
    outputs = data.outputs
    num_data_points = outputs.size()[0]
    dim_outputs = outputs.size()[1]

    return PreprocessedCalibrationData(
        inputs=data.inputs,
        outputs=data.outputs,
        std_noise=data.std_noise,
        num_data_points=num_data_points,
        dim_outputs=dim_outputs,
    )


def _validate_calibration_data(calibration_data: CalibrationData) -> None:
    inputs = calibration_data.inputs
    outputs = calibration_data.outputs
    if not inputs.size()[0] == outputs.size()[0]:
        raise UnvalidCalibrationDataError(
            "Size of input and output data does not match."
        )
