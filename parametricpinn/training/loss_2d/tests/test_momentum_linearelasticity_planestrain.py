import pytest
import torch
import torch.nn as nn

from parametricpinn.training.loss_2d import (
    momentum_equation_func_factory,
    traction_func_factory,
)
from parametricpinn.types import Module, Tensor


class FakeAnsatzPlaneStrain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._constants = torch.tensor([1 / 10, -2 / 5])

    def forward(self, x: Tensor) -> Tensor:
        coor_x = x[0]
        coor_y = x[1]
        return self._constants * torch.tensor(
            [coor_x**2 * coor_y, coor_x * coor_y**2]
        )


@pytest.fixture
def fake_ansatz() -> Module:
    return FakeAnsatzPlaneStrain()


def generate_volume_force(x_coordinates: Tensor, x_parameters: Tensor) -> Tensor:
    return torch.concat(
        (
            factor_volume_force * torch.unsqueeze(x_coordinates[:, 1], 1),
            2 * factor_volume_force * torch.unsqueeze(x_coordinates[:, 0], 1),
        ),
        dim=1,
    )


model = "plane strain"
youngs_modulus = 1.0
poissons_ratio = 0.25
factor_volume_force = 0.0


@pytest.mark.parametrize(
    ("x_coordinates", "x_parameters", "expected"),
    [
        (
            torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True),
            torch.tensor(
                [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
            ),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        ),
        (
            torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
            torch.tensor(
                [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
            ),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True),
            torch.tensor(
                [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
            ),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        ),
    ],
)
def test_momentum_equation_func(
    fake_ansatz: Module,
    x_coordinates: Tensor,
    x_parameters: Tensor,
    expected: Tensor,
):
    volume_forces = generate_volume_force(x_coordinates, x_parameters)
    sut = momentum_equation_func_factory(model=model)

    actual = sut(fake_ansatz, x_coordinates, x_parameters, volume_forces)

    torch.testing.assert_close(actual, expected)
