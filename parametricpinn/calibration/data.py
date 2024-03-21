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
    num_data_sets: int
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    std_noise: float


class CalibrationDataLoader2D:
    def __init__(
        self,
        input_subdir: str,
        num_cases: int,
        num_data_sets: int,
        num_data_points: int,
        std_noise: float,
        project_directory: ProjectDirectory,
        device: Device,
    ) -> None:
        self._input_subdir = input_subdir
        self._num_cases = num_cases
        self._num_data_sets = num_data_sets
        self._num_data_points = num_data_points
        self._std_noise = std_noise
        self._data_reader = CSVDataReader(project_directory)
        self._device = device
        self._file_name_data = "displacements"
        self._file_name_parameters = "parameters"
        self._slice_inputs = slice(0, 2)
        self._slice_outputs = slice(2, 4)

    def load_data(self) -> tuple[tuple[CalibrationData, ...], Tensor]:
        calibration_data_list: list[CalibrationData] = []
        true_parameters_list: list[Parameters] = []
        for case_index in range(self._num_cases):
            calibration_data, true_parameters = self._load_data_case(case_index)
            calibration_data_list.append(calibration_data)
            true_parameters_list.append(true_parameters)
        return tuple(calibration_data_list), torch.concat(true_parameters_list, dim=0)

    def _load_data_case(self, case_index: int) -> tuple[CalibrationData, Tensor]:
        inputs_sets, noisy_outputs_sets = self._load_data_sets(case_index)
        data = CalibrationData(
            num_data_sets=self._num_data_sets,
            inputs=inputs_sets,
            outputs=noisy_outputs_sets,
            std_noise=self._std_noise,
        )
        true_parameters = self._read_data(self._file_name_parameters, case_index)
        return data, true_parameters

    def _load_data_sets(
        self, case_index: int
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        data = self._read_data(self._file_name_data, case_index)
        return self._split_data_in_sets(data)

    def _read_data(self, file_name, idx_sample) -> Tensor:
        sample_subdir = f"sample_{idx_sample}"
        input_subdir = os.path.join(self._input_subdir, sample_subdir)
        data = self._data_reader.read(file_name, input_subdir)
        return torch.tensor(data, dtype=torch.get_default_dtype(), device=self._device)

    def _split_data_in_sets(
        self, data: Tensor
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        inputs_sets = []
        noisy_outputs_sets = []
        random_indices = self._generate_random_indices(len(data))
        for _ in range(self._num_data_sets):
            inputs_all = data[:, self._slice_inputs]
            inputs = inputs_all[random_indices, :]
            outputs_all = data[:, self._slice_outputs]
            outputs = outputs_all[random_indices, :]
            noisy_outputs = self._add_gaussian_noise(outputs)
            inputs_sets.append(inputs)
            noisy_outputs_sets.append(noisy_outputs)
        return tuple(inputs_sets), tuple(noisy_outputs_sets)

    def _generate_random_indices(self, max_index: int):
        return torch.randperm(max_index)[: self._num_data_points]

    def _add_gaussian_noise(self, outputs: Tensor) -> Tensor:
        return outputs + torch.normal(
            mean=0.0, std=self._std_noise, size=outputs.size(), device=self._device
        )


class CalibrationDataGenerator1D:
    def __init__(
        self,
        true_parameters: Parameters,
        traction: float,
        volume_force: float,
        length: float,
        num_cases: int,
        num_data_sets: int,
        num_data_points: int,
        std_noise: float,
        solution_func: LinearElasticDispalcementSolutionFunc,
        device: Device,
    ) -> None:
        self._true_parameters = true_parameters
        self._length = length
        self._traction = traction
        self._volume_force = volume_force
        self._num_cases = num_cases
        self._num_data_sets = num_data_sets
        self._num_data_points = num_data_points
        self._std_noise = std_noise
        self._solution_func = solution_func
        self._device = device

    def generate_data(self) -> tuple[CalibrationData, ...]:
        return tuple(
            self._generate_data_case(true_parameter)
            for true_parameter in self._true_parameters
        )

    def _generate_data_case(self, true_parameter: Parameters) -> CalibrationData:
        coordinates_sets = []
        noisy_displacements_sets = []
        for _ in range(self._num_data_sets):
            coordinates, noisy_displacements = self._generate_one_data_set(
                true_parameter
            )
            coordinates_sets.append(coordinates)
            noisy_displacements_sets.append(noisy_displacements)
        return CalibrationData(
            num_data_sets=self._num_data_sets,
            inputs=tuple(coordinates_sets),
            outputs=tuple(noisy_displacements_sets),
            std_noise=self._std_noise,
        )

    def _generate_one_data_set(self, true_parameter: Tensor) -> tuple[Tensor, Tensor]:
        coordinates = self._generate_random_coordinates()
        displacements = self._calculate_clean_displacements(coordinates, true_parameter)
        noisy_displacements = self._add_gaussian_noise(displacements)
        return coordinates, noisy_displacements

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

    def _add_gaussian_noise(self, displacements: Tensor) -> Tensor:
        return displacements + torch.normal(
            mean=0.0,
            std=self._std_noise,
            size=displacements.size(),
            device=self._device,
        )


@dataclass
class PreprocessedCalibrationData:
    num_data_sets: int
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    std_noise: float
    num_data_points_per_set: tuple[int, ...]
    num_total_data_points: int
    dim_outputs: int


def preprocess_calibration_data(data: CalibrationData) -> PreprocessedCalibrationData:
    _validate_calibration_data(data)
    outputs_sets = data.outputs
    num_data_points_per_set = tuple(len(outputs) for outputs in outputs_sets)
    num_total_data_points = sum(num_data_points_per_set)

    if outputs_sets[0].size() == torch.Size([num_data_points_per_set[0]]):
        dim_outputs = 1
    else:
        dim_outputs = outputs_sets[0].size()[1]

    return PreprocessedCalibrationData(
        num_data_sets=data.num_data_sets,
        inputs=data.inputs,
        outputs=data.outputs,
        std_noise=data.std_noise,
        num_data_points_per_set=num_data_points_per_set,
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )


@dataclass
class ConcatenatedCalibrationData:
    inputs: Tensor
    outputs: Tensor
    num_data_points: int
    dim_outputs: int
    std_noise: float


def concatenate_calibration_data(
    data: CalibrationData,
) -> ConcatenatedCalibrationData:
    _validate_calibration_data(data)
    num_data_sets = data.num_data_sets
    inputs_sets = data.inputs
    outputs_sets = data.outputs
    if num_data_sets == 1:
        concatenated_inputs = inputs_sets[0]
        concatenated_outputs = outputs_sets[0]
    else:
        concatenated_inputs = torch.concat(inputs_sets, dim=0)
        concatenated_outputs = torch.concat(outputs_sets, dim=0)
    num_total_data_points = len(concatenated_outputs)

    if concatenated_outputs.size() == torch.Size([num_total_data_points]):
        dim_outputs = 1
    else:
        dim_outputs = concatenated_outputs.size()[1]

    return ConcatenatedCalibrationData(
        inputs=concatenated_inputs,
        outputs=concatenated_outputs,
        num_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
        std_noise=data.std_noise,
    )


def _validate_calibration_data(calibration_data: CalibrationData) -> None:
    inputs_list = calibration_data.inputs
    outputs_list = calibration_data.outputs
    if not len(inputs_list) == len(outputs_list):
        raise UnvalidCalibrationDataError(
            "Size of input and output data sets does not match."
        )
    num_inputs = [input.size() for input in inputs_list]
    num_outputs = [output.size() for output in outputs_list]
    if not num_inputs == num_outputs:
        raise UnvalidCalibrationDataError(
            "Size of input and output data points does not match."
        )
