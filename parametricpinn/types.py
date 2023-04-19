# Standard library imports
from typing import TypeAlias

# Third-party imports
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

# Local library imports


## Pytorch
Tensor: TypeAlias = torch.Tensor
Parameter: TypeAlias = nn.Parameter
Module: TypeAlias = torch.nn.Module
