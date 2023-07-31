from dataclasses import dataclass

from parametricpinn.types import Tensor


@dataclass
class CalibrationData:
    inputs: Tensor
    outputs: Tensor
    std_noise: float


@dataclass
class PreprocessedData(CalibrationData):
    error_covariance_matrix: Tensor
    num_data_points: int
    dim_outputs: int