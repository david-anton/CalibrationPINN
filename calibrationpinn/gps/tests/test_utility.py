import pytest
import torch

from calibrationpinn.errors import UnvalidGPParametersError
from calibrationpinn.gps.utility import validate_parameters_size
from calibrationpinn.types import Tensor, TensorSize


@pytest.mark.parametrize(
    ("parameters", "valid_size"),
    [
        (torch.tensor([1]), 1),
        (torch.tensor([1]), torch.Size([1])),
        (torch.tensor([1, 1]), torch.Size([2])),
        (torch.tensor([[1, 1], [1, 1]]), torch.Size([2, 2])),
    ],
)
def test_validate_parameters_size_for_valid_parameters(
    parameters: Tensor, valid_size: int | TensorSize
) -> None:
    sut = validate_parameters_size
    try:
        sut(parameters=parameters, valid_size=valid_size)
    except UnvalidGPParametersError:
        assert False


@pytest.mark.parametrize(
    ("parameters", "valid_size"),
    [
        (torch.tensor([1]), 2),
        (torch.tensor([1]), torch.Size([2])),
        (torch.tensor([1, 1]), torch.Size([1])),
        (torch.tensor([[1, 1], [1, 1]]), torch.Size([1, 1])),
    ],
)
def test_validate_parameters_size_for_unvalid_parameters(
    parameters: Tensor, valid_size: int | TensorSize
) -> None:
    sut = validate_parameters_size
    with pytest.raises(UnvalidGPParametersError):
        sut(parameters=parameters, valid_size=valid_size)
