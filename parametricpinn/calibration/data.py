import os
import random
from dataclasses import dataclass
from typing import TypeAlias

import torch

from parametricpinn.data.validationdata_linearelasticity_1d import (
    LinearElasticDispalcementSolutionFunc,
)
from parametricpinn.errors import UnvalidCalibrationDataError
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.types import Device, Tensor

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
        device: Device,
    ) -> None:
        self._input_subdir = input_subdir
        self._num_data_points = num_data_points
        self._std_noise = std_noise
        self._num_cases = num_cases
        self._data_reader = CSVDataReader(project_directory)
        self._device = device
        self._file_name_displacements = "displacements"
        self._file_name_parameters = "parameters"
        self._slice_coordinates = slice(0, 2)
        self._slice_displacements = slice(2, 4)

    def load_data(self) -> tuple[tuple[CalibrationData, ...], Tensor]:
        data_sets_list: list[CalibrationData] = []
        true_parameters_list: list[Parameters] = []
        for idx_case in range(self._num_cases):
            displacements = self._read_data(self._file_name_displacements, idx_case)
            true_parameters = self._read_data(self._file_name_parameters, idx_case)
            random_indices = self._generate_random_indices(displacements.size(dim=0))
            data_set = self._create_data_set(displacements, random_indices)
            data_sets_list.append(data_set)
            true_parameters_list.append(true_parameters)
        return tuple(data_sets_list), torch.concat(true_parameters_list, dim=0)

    def _read_data(self, file_name, idx_sample) -> Tensor:
        sample_subdir = f"sample_{idx_sample}"
        input_subdir = os.path.join(self._input_subdir, sample_subdir)
        data = self._data_reader.read(file_name, input_subdir)
        return torch.tensor(data, dtype=torch.get_default_dtype(), device=self._device)

    def _generate_random_indices(self, max_index: int):
        return torch.randperm(max_index)[: self._num_data_points]

    def _create_data_set(
        self, displacements: Tensor, random_indices: Tensor
    ) -> CalibrationData:
        inputs_all = displacements[:, self._slice_coordinates]
        inputs = inputs_all[random_indices, :]
        outputs_all = displacements[:, self._slice_displacements]
        outputs = outputs_all[random_indices, :]
        noisy_outputs = outputs + torch.normal(
            mean=0.0, std=self._std_noise, size=outputs.size(), device=self._device
        )
        return CalibrationData(
            inputs=inputs, outputs=noisy_outputs, std_noise=self._std_noise
        )


class CalibrationDataGenerator1D:
    def __init__(
        self,
        true_parameters: Parameters,
        traction: float,
        volume_force: float,
        length: float,
        num_data_points: int,
        std_noise: float,
        num_cases: int,
        solution_func: LinearElasticDispalcementSolutionFunc,
        device: Device,
    ) -> None:
        self._true_parameters = true_parameters
        self._length = length
        self._traction = traction
        self._volume_force = volume_force
        self._num_data_points = num_data_points
        self._std_noise = std_noise
        self._num_cases = num_cases
        self._solution_func = solution_func
        self._device = device

    def generate_data(self) -> tuple[CalibrationData, ...]:
        return tuple(
            self._generate_data_set(true_parameter)
            for true_parameter in self._true_parameters
        )

    def _generate_data_set(self, true_parameter: Parameters) -> CalibrationData:
        coordinates = self._generate_random_coordinates()
        displacements = self._calculate_clean_displacements(coordinates, true_parameter)
        noisy_displacements = displacements + torch.normal(
            mean=0.0,
            std=self._std_noise,
            size=displacements.size(),
            device=self._device,
        )
        return CalibrationData(
            inputs=coordinates, outputs=noisy_displacements, std_noise=self._std_noise
        )

    def _generate_random_coordinates(self) -> Tensor:
        normalized_length = torch.rand(
            size=(self._num_data_points, 1), requires_grad=True, device=self._device
        )
        return self._length * normalized_length

    def _calculate_clean_displacements(
        self, coordinates: Tensor, true_parameter: Tensor
    ) -> Tensor:
        return (
            self._solution_func(
                coordinates,
                self._length,
                true_parameter,
                self._traction,
                self._volume_force,
            )
            .clone()
            .detach()
        )


@dataclass
class PreprocessedCalibrationData(CalibrationData):
    num_data_points: int
    dim_outputs: int


def preprocess_calibration_data(data: CalibrationData) -> PreprocessedCalibrationData:
    _validate_calibration_data(data)
    outputs = data.outputs
    num_data_points = outputs.size()[0]
    if outputs.size() == torch.Size([num_data_points]):
        dim_outputs = 1
    else:
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
