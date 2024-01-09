from collections import namedtuple
from typing import Callable, TypeAlias

from parametricpinn.types import Tensor

TrainingData1DCollocation = namedtuple(
    "TrainingData1DCollocation", ["x_coor", "x_E", "f", "y_true"]
)
TrainingData1DTractionBC = namedtuple(
    "TrainingData1DTractionBC", ["x_coor", "x_E", "y_true"]
)

TrainingData2DCollocation = namedtuple(
    "TrainingData2DCollocation", ["x_coor", "x_E", "x_nu", "f"]
)
TrainingData2DTractionBC = namedtuple(
    "TrainingData2DTractionBC",
    ["x_coor", "x_E", "x_nu", "normal", "area_frac", "y_true"],
)
TrainingData2DStressBC = namedtuple("TrainingData2DStressBC", ["x_coor", "x_E", "x_nu"])
TrainingData2DSymmetryBC = namedtuple(
    "TrainingData2DSymmetryBC", ["x_coor_1", "x_coor_2", "x_E", "x_nu"]
)

ValidationBatch: TypeAlias = tuple[Tensor, Tensor]
ValidationBatchList: TypeAlias = list[ValidationBatch]
ValidationCollateFunc: TypeAlias = Callable[[ValidationBatchList], ValidationBatch]
