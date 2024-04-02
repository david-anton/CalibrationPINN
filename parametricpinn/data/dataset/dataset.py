from collections import namedtuple
from typing import Callable, TypeAlias

from parametricpinn.types import Tensor

TrainingData1DCollocation = namedtuple(
    "TrainingData1DCollocation", ["x_coor", "x_params", "f"]
)
TrainingData1DTractionBC = namedtuple(
    "TrainingData1DTractionBC", ["x_coor", "x_params", "y_true"]
)

TrainingData2DCollocation = namedtuple(
    "TrainingData2DCollocation", ["x_coor", "x_params", "f"]
)
TrainingData2DTractionBC = namedtuple(
    "TrainingData2DTractionBC",
    ["x_coor", "x_params", "normal", "area_frac", "y_true"],
)
TrainingData2DStressBC = namedtuple("TrainingData2DStressBC", ["x_coor", "x_params"])
TrainingData2DSymmetryBC = namedtuple(
    "TrainingData2DSymmetryBC", ["x_coor_1", "x_coor_2", "x_params"]
)

SimulationBatch: TypeAlias = tuple[Tensor, Tensor]
SimulationBatchList: TypeAlias = list[SimulationBatch]
SimulationCollateFunc: TypeAlias = Callable[[SimulationBatchList], SimulationBatch]
