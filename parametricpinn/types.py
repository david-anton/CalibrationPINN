from dataclasses import dataclass
from typing import Protocol, TypeAlias, Union

import gpytorch
import numpy as np
import numpy.typing as npt
import pandas as pd
import shapely
import torch
import torch.nn as nn
from matplotlib import axes, figure
from scipy.stats._continuous_distns import norm_gen
from scipy.stats._multivariate import multivariate_normal_gen

# Pytorch
Tensor: TypeAlias = torch.Tensor
TensorSize: TypeAlias = torch.Size
Parameter: TypeAlias = nn.Parameter
Module: TypeAlias = torch.nn.Module
Device: TypeAlias = torch.device
TorchUniformDist: TypeAlias = torch.distributions.Uniform
TorchUniNormalDist: TypeAlias = torch.distributions.Normal
TorchMultiNormalDist: TypeAlias = torch.distributions.MultivariateNormal

# GPyTorch
GPModel: TypeAlias = gpytorch.models.ExactGP
GPMean: TypeAlias = gpytorch.means.Mean
GPKernel: TypeAlias = gpytorch.kernels.Kernel

# Numpy
NPArray: TypeAlias = Union[
    npt.NDArray[np.int16],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
]

# Scipy statistics
SciPyUniNormalDist: TypeAlias = norm_gen
SciPyMultiNormalDist: TypeAlias = multivariate_normal_gen

# Pandas
PDDataFrame: TypeAlias = pd.DataFrame

# Matplotlib
PLTFigure: TypeAlias = figure.Figure
PLTAxes: TypeAlias = axes.Axes

# Shapely
ShapelyPolygon: TypeAlias = shapely.Polygon


# dataclass
@dataclass
class DataClass(Protocol):
    pass
