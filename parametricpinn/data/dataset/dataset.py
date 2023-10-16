from collections import namedtuple
from typing import Callable, NamedTuple, Protocol, TypeAlias, Union

from parametricpinn.data.dataset.platewithholedatasets import (
    TrainingCollateFunc as PlateWithHoleTrainingCollateFunc,
)
from parametricpinn.data.dataset.quarterplatewithholedatasets import (
    TrainingCollateFunc as QuarterPlateWithHoleTrainingCollateFunc,
)
from parametricpinn.types import Tensor

TrainingData: TypeAlias = NamedTuple
ValidationBatch: TypeAlias = tuple[Tensor, Tensor]
ValidationBatchList: TypeAlias = list[ValidationBatch]
ValidationCollateFunc: TypeAlias = Callable[[ValidationBatchList], ValidationBatch]
TrainingCollateFunc: TypeAlias = Union[
    QuarterPlateWithHoleTrainingCollateFunc, PlateWithHoleTrainingCollateFunc
]


TrainingData2DCollocation = namedtuple(
    "TrainingData2DCollocation", ["x_coor", "x_E", "x_nu", "f"]
)
TrainingData2DSymmetryBC = namedtuple(
    "TrainingData2DSymmetryBC", ["x_coor", "x_E", "x_nu"]
)
TrainingData2DTractionBC = namedtuple(
    "TrainingData2DTractionBC",
    ["x_coor", "x_E", "x_nu", "normal", "area_frac", "y_true"],
)


class TrainingDataset(Protocol):
    def get_collate_func(self) -> TrainingCollateFunc:
        pass


class ValidationDataset(Protocol):
    def get_collate_func(self) -> ValidationCollateFunc:
        pass
