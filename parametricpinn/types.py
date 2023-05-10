from typing import TypeAlias, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


## Pytorch
Tensor: TypeAlias = torch.Tensor
Parameter: TypeAlias = nn.Parameter
Module: TypeAlias = torch.nn.Module

## Numpy
NPArray: TypeAlias = Union[
    npt.NDArray[np.int16],
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float64],
]
