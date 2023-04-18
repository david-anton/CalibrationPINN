# Standard library imports
import random

# Third-party imports
import numpy as np
import torch

# Local library imports


def set_default_dtype(dtype: torch.dtype) -> None:
    torch.set_default_dtype(dtype)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device")
    return device


class Settings:
    def __init__(self) -> None:
        self.PROJECT_DIRECTORY_PATH = Path(
            os.getenv("APP_HOME", os.getenv("HOME", "."))
        )
        self.OUTPUT_SUBDIRECTORY_NAME = "output"
        self.INPUT_SUBDIRECTORY_NAME = "input"
