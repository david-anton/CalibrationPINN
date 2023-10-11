import pytest
import torch

from parametricpinn.training.loss_2d.momentum_neohookean import (
    _calculate_determinant,
    _calculate_first_lame_constant_lambda,
    _calculate_second_lame_constant_mu,
    _deformation_gradient_func,
    _first_piola_stress_tensor_func,
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


def calculate_first_lame_constant_lambda(x_parameters: Tensor) -> Tensor:
    E = x_parameters[0]
    nu = x_parameters[1]
    lambda_ = (E * nu) / ((1.0 - 2.0 * nu) * (1.0 + nu))
    return torch.unsqueeze(lambda_, dim=0)


def calculate_second_lame_constant_mu(x_parameters: Tensor) -> Tensor:
    E = x_parameters[0]
    nu = x_parameters[1]
    mu_ = E / (2.0 * (1.0 + nu))
    return torch.unsqueeze(mu_, dim=0)


# def calculate_free_energy(deformation_gradient: Tensor, x_param: Tensor) -> Tensor:
#     # Plane stress assumed
#     F = deformation_gradient
#     J = torch.det(F)
#     C = torch.matmul(F.T, F)
#     I_c = torch.trace(C)
#     param_lambda = calculate_first_lame_constant_lambda(x_param)
#     param_mu = calculate_second_lame_constant_mu(x_param)
#     param_C = param_mu / 2
#     param_D = param_lambda / 2
#     free_energy = param_C * (I_c - 2 - 2 * torch.log(J)) + param_D * (J - 1) ** 2
#     return torch.squeeze(free_energy, 0)


def calculate_first_piola_stress_tensor(
    deformation_gradient: Tensor, x_parameters: Tensor
) -> Tensor:
    # Plane stress assumed
    F = deformation_gradient
    T_inv_F = torch.transpose(torch.inverse(F), 0, 1)
    det_F = _calculate_determinant(F)
    param_lambda = calculate_first_lame_constant_lambda(x_parameters)
    param_mu = calculate_second_lame_constant_mu(x_parameters)
    param_C = param_mu / 2
    param_D = param_lambda / 2
    return 2 * param_C * (F - T_inv_F) + 2 * param_D * (det_F - 1) * T_inv_F


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


@pytest.mark.parametrize(
    ("x_parameters"),
    [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 1.0]),
    ],
)
def test_calculate_first_lame_constant_lambda(
    x_parameters: Tensor,
) -> None:
    sut = _calculate_first_lame_constant_lambda

    actual = sut(x_parameters)

    expected = calculate_first_lame_constant_lambda(x_parameters)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_parameters"),
    [
        torch.tensor([0.0, 0.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 1.0]),
    ],
)
def test_calculate_second_lame_constant_mu(
    x_parameters: Tensor,
) -> None:
    sut = _calculate_second_lame_constant_mu

    actual = sut(x_parameters)

    expected = calculate_second_lame_constant_mu(x_parameters)
    torch.testing.assert_close(actual, expected)


# def test_calculate_right_cuachy_green_tensor() -> None:
#     a = 1.0
#     b = 2.0
#     c = 3.0
#     d = 4.0
#     deformation_gradient = torch.tensor([[a, b], [c, d]])
#     sut = _calculate_right_cuachy_green_tensor

#     actual = sut(deformation_gradient)

#     expected = torch.tensor(
#         [[a * a + c * c, b * a + d * c], [a * b + c * d, b * b + d * d]]
#     )
#     torch.testing.assert_close(actual, expected)


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


# def test_calculate_first_invariant() -> None:
#     a = 1.0
#     b = 2.0
#     c = 3.0
#     d = 4.0
#     tensor = torch.tensor([[a, b], [c, d]])
#     sut = _calculate_first_invariant

#     actual = sut(tensor)

#     expected = torch.tensor([a + d])
#     torch.testing.assert_close(actual, expected)


# @pytest.mark.parametrize(
#     ("deformation_gradient", "x_parameters"),
#     [
#         (
#             torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
#             torch.tensor([1.0, 1.0]),
#         ),
#         (
#             torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
#             torch.tensor([1.0, 1.0]),
#         ),
#         (
#             torch.tensor([[2.0, 1.0], [0.0, 1.0]]),
#             torch.tensor([1.0, 1.0]),
#         ),
#     ],
# )
# def test_free_energy_func(deformation_gradient: Tensor, x_parameters: Tensor) -> None:
#     sut = _free_energy_func

#     actual = sut(deformation_gradient, x_parameters)
#     expected = calculate_free_energy(deformation_gradient, x_parameters)
#     torch.testing.assert_close(actual, expected)


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
