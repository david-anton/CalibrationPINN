# Standard library imports
from typing import Union, TypeAlias

# Third-party imports
import numpy as np
import numpy.typing as npt
import torch

# Local library imports


## Pytorch
Tensor: TypeAlias = torch.Tensor
Module: TypeAlias = torch.nn.Module

## Numpy
# NPArray: TypeAlias = Union[
#     npt.NDArray[np.int16],
#     npt.NDArray[np.int32],
#     npt.NDArray[np.float32],
#     npt.NDArray[np.float64],
# ]
