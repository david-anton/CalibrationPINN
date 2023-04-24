import pytest
import torch
import torch.nn as nn

from parametricpinn.training.loss_1d import momentum_equation_func, stress_func
from parametricpinn.types import Module, Tensor

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
    ("x_coordinate", "x_youngs_modulus", "volume_force", "expected"),
    [
        (
            torch.tensor([[-1.0], [-1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus], [youngs_modulus]]),
            torch.tensor([[volume_force], [volume_force]]),
            torch.tensor([[0.0], [0.0]]),
        ),
        (
            torch.tensor([[0.0], [0.0]], requires_grad=True),
            torch.tensor([[youngs_modulus], [youngs_modulus]]),
            torch.tensor([[volume_force], [volume_force]]),
            torch.tensor([[0.0], [0.0]]),
        ),
        (
            torch.tensor([[1.0], [1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus], [youngs_modulus]]),
            torch.tensor([[volume_force], [volume_force]]),
            torch.tensor([[0.0], [0.0]]),
        ),
    ],
)
def test_momentum_equation_func(
    fake_ansatz: Module,
    x_coordinate: Tensor,
    x_youngs_modulus: Tensor,
    volume_force: Tensor,
    expected: Tensor,
):
    sut = momentum_equation_func

    actual = sut(
        ansatz=fake_ansatz,
        x_coor=x_coordinate,
        x_param=x_youngs_modulus,
        volume_force=volume_force,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_coordinate", "x_youngs_modulus", "expected"),
    [
        (
            torch.tensor([[-1.0], [-1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus], [youngs_modulus]]),
            torch.tensor([[2.0], [2.0]]),
        ),
        (
            torch.tensor([[0.0], [0.0]], requires_grad=True),
            torch.tensor([[youngs_modulus], [youngs_modulus]]),
            torch.tensor([[0.0], [0.0]]),
        ),
        (
            torch.tensor([[1.0], [1.0]], requires_grad=True),
            torch.tensor([[youngs_modulus], [youngs_modulus]]),
            torch.tensor([[-2.0], [-2.0]]),
        ),
    ],
)
def test_stress_func(
    fake_ansatz: Module,
    x_coordinate: Tensor,
    x_youngs_modulus: Tensor,
    expected: Tensor,
):
    sut = stress_func

    actual = sut(ansatz=fake_ansatz, x_coor=x_coordinate, x_param=x_youngs_modulus)
    torch.testing.assert_close(actual, expected)
