import os
from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
from torch.utils.data import Dataset

from parametricpinn.data.base import repeat_tensor
from parametricpinn.data.dataset.dataset import (
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
    ValidationBatch,
    ValidationBatchList,
    ValidationCollateFunc,
)
from parametricpinn.data.geometry import QuarterPlateWithHole
from parametricpinn.io import ProjectDirectory
from parametricpinn.io.readerswriters import CSVDataReader
from parametricpinn.types import Tensor

TrainingBatch: TypeAlias = tuple[
    TrainingData2DCollocation,
    TrainingData2DTractionBC,
]
TrainingBatchList: TypeAlias = list[TrainingBatch]
TrainingCollateFunc: TypeAlias = Callable[[TrainingBatchList], TrainingBatch]
