# Standard library imports

# Third-party imports
import pytest
import torch
import torch.nn as nn

# Local library imports
from parametricpinn.training.loss import stress_func_1D, momentum_equation_func_1D
from parametricpinn.types import Tensor, Module

youngs_modulus = 1.0
volume_force = 2.0


class FakeAnsatz(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return -(1 / 2 * x**2 * volume_force)


@pytest.fixture
def fake_ansatz() -> Module:
    return FakeAnsatz()


@pytest.mark.parametrize(
    ("x_coordinate", "x_youngs_modulus", "expected"),
    [
        (
            torch.tensor([[-1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus]]),
            torch.tensor([[2.0]]),
        ),
        (
            torch.tensor([[0.0]], requires_grad=True),
            torch.tensor([[youngs_modulus]]),
            torch.tensor([[0.0]]),
        ),
        (
            torch.tensor([[1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus]]),
            torch.tensor([[-2.0]]),
        ),
    ],
)
def test_stress_func_1D(
    fake_ansatz: Module,
    x_coordinate: Tensor,
    x_youngs_modulus: Tensor,
    expected: Tensor,
):
    sut = stress_func_1D

    actual = sut(ansatz=fake_ansatz, x_coor=x_coordinate, x_E=x_youngs_modulus)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_coordinate", "x_youngs_modulus", "expected"),
    [
        (
            torch.tensor([[-1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus]]),
            torch.tensor([[0.0]]),
        ),
        (
            torch.tensor([[0.0]], requires_grad=True),
            torch.tensor([[youngs_modulus]]),
            torch.tensor([[0.0]]),
        ),
        (
            torch.tensor([[1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus]]),
            torch.tensor([[0.0]]),
        ),
    ],
)
def test_momentum_equation_func_1D(
    fake_ansatz: Module,
    x_coordinate: Tensor,
    x_youngs_modulus: Tensor,
    expected: Tensor,
):
    sut = momentum_equation_func_1D

    actual = sut(
        ansatz=fake_ansatz,
        x_coor=x_coordinate,
        x_E=x_youngs_modulus,
        volume_force=torch.tensor(volume_force),
    )
    torch.testing.assert_close(actual, expected)
