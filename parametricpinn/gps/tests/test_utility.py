import pytest
import torch

from parametricpinn.errors import UnvalidGPParametersError
from parametricpinn.gps.utility import validate_parameters_size
from parametricpinn.types import Tensor, TensorSize


@pytest.mark.parametrize(
    ("parameters", "valid_size"),
    [
        (torch.tensor([1]), torch.Size([1])),
        (torch.tensor([1, 1]), torch.Size([2])),
        (torch.tensor([[1, 1], [1, 1]]), torch.Size([2, 2])),
    ],
)
def test_validate_parameters_size_for_valid_parameters(
    parameters: Tensor, valid_size: TensorSize
) -> None:
    sut = validate_parameters_size
    try:
        sut(parameters=parameters, valid_size=valid_size)
    except UnvalidGPParametersError:
        assert False


@pytest.mark.parametrize(
    ("parameters", "valid_size"),
    [
        (torch.tensor([1]), torch.Size([2])),
        (torch.tensor([1, 1]), torch.Size([1])),
        (torch.tensor([[1, 1], [1, 1]]), torch.Size([1, 1])),
    ],
)
def test_validate_parameters_size_for_unvalid_parameters(
    parameters: Tensor, valid_size: TensorSize
) -> None:
    sut = validate_parameters_size
    with pytest.raises(UnvalidGPParametersError):
        sut(parameters=parameters, valid_size=valid_size)
