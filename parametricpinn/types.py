from dataclasses import dataclass
from typing import Protocol, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import axes, figure
from scipy.stats._continuous_distns import norm_gen
from scipy.stats._multivariate import multivariate_normal_gen

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

# Scipy statistics
UniNormalDist: TypeAlias = norm_gen
MultiNormalDist: TypeAlias = multivariate_normal_gen

# Pandas
PDDataFrame: TypeAlias = pd.DataFrame

# Matplotlib
PLTFigure: TypeAlias = figure.Figure
PLTAxes: TypeAlias = axes.Axes


# dataclass
@dataclass
class DataClass(Protocol):
    pass
