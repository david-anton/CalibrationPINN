import pytest
import torch

from calibrationpinn.calibration.data import (
    CalibrationData,
    ConcatenatedCalibrationData,
    PreprocessedCalibrationData,
    concatenate_calibration_data,
    preprocess_calibration_data,
)

dim_outputs = 1
num_data_sets = 2
num_data_points_in_one_set = 3
num_data_points_per_set = (num_data_points_in_one_set, num_data_points_in_one_set)
num_total_data_points = num_data_sets * num_data_points_in_one_set
inputs_set_1 = torch.tensor([[1.0], [1.1], [1.2]])
inputs_set_2 = torch.tensor([[2.0], [2.1], [2.2]])
outputs_set_1 = torch.tensor([[10.0], [10.1], [10.2]])
outputs_set_2 = torch.tensor([[20.0], [20.1], [20.2]])
inputs = (inputs_set_1, inputs_set_2)
outputs = (outputs_set_1, outputs_set_2)
concatenated_inputs = torch.concatenate((inputs_set_1, inputs_set_2), dim=0)
concatenated_outputs = torch.concatenate((outputs_set_1, outputs_set_2), dim=0)
outputs = (outputs_set_1, outputs_set_2)
std_noise = 4


@pytest.fixture
def calibration_data() -> CalibrationData:
    return CalibrationData(
        num_data_sets=num_data_sets, inputs=inputs, outputs=outputs, std_noise=std_noise
    )


### Preprocessed calibration data


def test_preprocessed_calibration_data_number_data_sets(
    calibration_data: CalibrationData,
) -> None:
    sut = preprocess_calibration_data(calibration_data)

    actual = sut.num_data_sets
    expected = num_data_sets
    assert actual == expected


def test_preprocessed_calibration_data_inputs(
    calibration_data: CalibrationData,
) -> None:
    sut = preprocess_calibration_data(calibration_data)

    actual = sut.inputs
    expected = inputs
    torch.testing.assert_close(actual, expected)


def test_preprocessed_calibration_data_outputs(
    calibration_data: CalibrationData,
) -> None:
    sut = preprocess_calibration_data(calibration_data)

    actual = sut.outputs
    expected = outputs
    torch.testing.assert_close(actual, expected)


def test_preprocessed_calibration_data_standard_deviation_noise(
    calibration_data: CalibrationData,
) -> None:
    sut = preprocess_calibration_data(calibration_data)

    actual = sut.std_noise
    expected = std_noise
    assert actual == expected


def test_preprocessed_calibration_data_number_data_points_per_set(
    calibration_data: CalibrationData,
) -> None:
    sut = preprocess_calibration_data(calibration_data)

    actual = sut.num_data_points_per_set
    expected = num_data_points_per_set
    assert actual == expected


def test_preprocessed_calibration_data_number_total_data_points(
    calibration_data: CalibrationData,
) -> None:
    sut = preprocess_calibration_data(calibration_data)

    actual = sut.num_total_data_points
    expected = num_total_data_points
    assert actual == expected


def test_preprocessed_calibration_data_dimensions_output(
    calibration_data: CalibrationData,
) -> None:
    sut = preprocess_calibration_data(calibration_data)

    actual = sut.dim_outputs
    expected = dim_outputs
    assert actual == expected


### Concatenated calibration data


def test_concatenated_calibration_data_inputs(
    calibration_data: CalibrationData,
) -> None:
    sut = concatenate_calibration_data(calibration_data)

    actual = sut.inputs
    expected = concatenated_inputs
    torch.testing.assert_close(actual, expected)


def test_concatenated_calibration_data_outputs(
    calibration_data: CalibrationData,
) -> None:
    sut = concatenate_calibration_data(calibration_data)

    actual = sut.outputs
    expected = concatenated_outputs
    torch.testing.assert_close(actual, expected)


def test_concatenated_calibration_number_data_points(
    calibration_data: CalibrationData,
) -> None:
    sut = concatenate_calibration_data(calibration_data)

    actual = sut.num_data_points
    expected = num_total_data_points
    torch.testing.assert_close(actual, expected)


def test_concatenated_calibration_data_dimensions_output(
    calibration_data: CalibrationData,
) -> None:
    sut = concatenate_calibration_data(calibration_data)

    actual = sut.dim_outputs
    expected = dim_outputs
    assert actual == expected


def test_concatenated_calibration_data_standard_deviation_noise(
    calibration_data: CalibrationData,
) -> None:
    sut = concatenate_calibration_data(calibration_data)

    actual = sut.std_noise
    expected = std_noise
    assert actual == expected
