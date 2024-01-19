import pytest
import torch

from parametricpinn.training.loss_2d.momentum_neohookean import (
    _calculate_determinant,
    _deformation_gradient_func,
    _first_piola_stress_tensor_func,
    _calculate_right_cauchy_green_tensor,
    _calculate_first_invariant,
)
from parametricpinn.training.loss_2d.tests.testdoubles import TransformedFakeAnsatz
from parametricpinn.types import Module, Tensor

constant_displacement_x = 1.0
constant_displacement_y = 1.0


@pytest.fixture
def fake_ansatz() -> Module:
    return TransformedFakeAnsatz(constant_displacement_x, constant_displacement_y)


def calculate_single_deformation_gradient(x_coordinates: Tensor) -> Tensor:
    coordinate_x = x_coordinates[0]
    coordinate_y = x_coordinates[1]
    identity = torch.eye(n=2)
    jacobian_displacement = torch.tensor(
        [
            [2 * coordinate_x * coordinate_y, coordinate_x**2],
            [coordinate_y**2, 2 * coordinate_x * coordinate_y],
        ]
    )
    return jacobian_displacement + identity


def calculate_first_piola_stress_tensor(
    deformation_gradient: Tensor, x_parameters: Tensor
) -> Tensor:
    # Plane stress assumed ???
    # Identity
    I = torch.eye(n=2)
    # Deformation gradient
    F = deformation_gradient
    J = torch.unsqueeze(torch.det(F), dim=0)

    # Unimodular deformation tensors
    uni_F = (J ** (-1 / 3)) * F  # unimodular deformation gradient
    transpose_uni_F = torch.transpose(uni_F, 0, 1)
    uni_C = torch.matmul(transpose_uni_F, uni_F)  # unimodular right Cauchy-Green tensor

    # Invariants of unimodular deformation tensors
    uni_I_c = torch.trace(uni_C)

    # Material parameters
    param_K = x_parameters[0]
    param_c_10 = x_parameters[1]

    # 2. Piola-Kirchoff stress tensor
    inv_uni_C = torch.inverse(uni_C)
    T = J * param_K * (J - 1) * inv_uni_C + 2 * (J ** (-2 / 3)) * (
        param_c_10 * I - (1 / 3) * param_c_10 * uni_I_c * inv_uni_C
    )

    # 1. Piola-Kirchoff stress tensor
    P = F * T
    return P


@pytest.mark.parametrize(
    ("x_coordinates"),
    [
        torch.tensor([2.0, 2.0], requires_grad=True),
        torch.tensor([0.0, 0.0], requires_grad=True),
        torch.tensor([-2.0, -2.0], requires_grad=True),
    ],
)
def test_deformation_gradient_func(
    fake_ansatz: Module,
    x_coordinates: Tensor,
) -> None:
    x_parameters = torch.tensor([0.0, 0.0], requires_grad=True)

    sut = _deformation_gradient_func

    actual = sut(fake_ansatz, x_coordinates, x_parameters)

    expected = calculate_single_deformation_gradient(x_coordinates)
    torch.testing.assert_close(actual, expected)


def test_calculate_determinant() -> None:
    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0
    tensor = torch.tensor([[a, b], [c, d]])
    sut = _calculate_determinant

    actual = sut(tensor)

    expected = torch.tensor([a * d - b * c])
    torch.testing.assert_close(actual, expected)


def test_calculate_right_cuachy_green_tensor() -> None:
    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0
    deformation_gradient = torch.tensor([[a, b], [c, d]])
    sut = _calculate_right_cauchy_green_tensor

    actual = sut(deformation_gradient)

    expected = torch.tensor(
        [[a * a + c * c, a * b + c * d], [b * a + d * c, b * b + d * d]]
    )
    torch.testing.assert_close(actual, expected)


def test_calculate_first_invariant() -> None:
    a = 1.0
    b = 2.0
    c = 3.0
    d = 4.0
    tensor = torch.tensor([[a, b], [c, d]])
    sut = _calculate_first_invariant

    actual = sut(tensor)

    expected = torch.tensor([a + d])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_coordinates", "x_parameters"),
    [
        (
            torch.tensor([2.0, 2.0]),
            torch.tensor([1.0, 2.0]),
        ),
        (
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 2.0]),
        ),
        (
            torch.tensor([-2.0, -2.0]),
            torch.tensor([1.0, 2.0]),
        ),
    ],
)
def test_first_piola_stress_tensor_func(
    fake_ansatz: Module, x_coordinates: Tensor, x_parameters: Tensor
) -> None:
    sut = _first_piola_stress_tensor_func

    actual = sut(fake_ansatz, x_coordinates, x_parameters)

    deformation_gradient = calculate_single_deformation_gradient(x_coordinates)
    expected = calculate_first_piola_stress_tensor(deformation_gradient, x_parameters)
    torch.testing.assert_close(actual, expected)
