from dataclasses import dataclass
from typing import Protocol, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import figure

# Pytorch
Tensor: TypeAlias = torch.Tensor
Parameter: TypeAlias = nn.Parameter
Module: TypeAlias = torch.nn.Module
Device: TypeAlias = torch.device

# Numpy
NPArray: TypeAlias = Union[
    npt.NDArray[np.int16],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
]

# Pandas
PDDataFrame: TypeAlias = pd.DataFrame

# Matplotlib
PLTFigure: TypeAlias = figure.Figure


# dataclass
@dataclass
class DataClass(Protocol):
    pass
