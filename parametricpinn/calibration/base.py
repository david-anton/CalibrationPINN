from dataclasses import dataclass
from typing import TypeAlias

from parametricpinn.errors import UnvalidCalibrationDataError
from parametricpinn.types import Tensor

Parameters: TypeAlias = Tensor


@dataclass
class CalibrationData:
    inputs: Tensor
    outputs: Tensor
    std_noise: float


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
