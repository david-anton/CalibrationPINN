import pytest
import torch

from parametricpinn.training.loss_2d.momentum_neohooke import (
    _calculate_determinant,
    _calculate_first_invariant,
    _calculate_right_cauchy_green_tensor,
    _deformation_gradient_func,
    cauchy_stress_func_factory,
    first_piola_kirchhoff_stress_func_factory,
    traction_func_factory,
)
from parametricpinn.training.loss_2d.tests.testdoubles import (
    FakeAnsatz,
    TransformedFakeAnsatz,
)
from parametricpinn.types import Module, Tensor

constant_displacement_x = 1.0
constant_displacement_y = 1.0


@pytest.fixture
def fake_ansatz() -> Module:
    return FakeAnsatz(constant_displacement_x, constant_displacement_y)


@pytest.fixture
def transformed_fake_ansatz() -> Module:
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


def calculate_first_piola_kirchhof_stress(
    x_coordinates_list: Tensor, x_parameters_list: Tensor
) -> Tensor:
    return torch.stack(
        [
            _calculate_single_first_piola_kirchhoff_stress_tensor(
                x_coordinates, x_parameters
            )
            for (x_coordinates, x_parameters) in zip(
                torch.unbind(x_coordinates_list),
                torch.unbind(x_parameters_list),
            )
        ]
    )


def calculate_cauchy_stress(
    x_coordinates_list: Tensor, x_parameters_list: Tensor
) -> Tensor:
    return torch.stack(
        [
            _calculate_single_cauchy_stress_tensor(x_coordinates, x_parameters)
            for (x_coordinates, x_parameters) in zip(
                torch.unbind(x_coordinates_list),
                torch.unbind(x_parameters_list),
            )
        ]
    )


def calculate_traction(
    x_coordinates_list: Tensor, x_parameters_list: Tensor, normal_vector_list: Tensor
) -> Tensor:
    return torch.stack(
        [
            _calculate_single_traction(x_coordinates, x_parameters, normal_vector)
            for (x_coordinates, x_parameters, normal_vector) in zip(
                torch.unbind(x_coordinates_list),
                torch.unbind(x_parameters_list),
                torch.unbind(normal_vector_list),
            )
        ]
    )


def _calculate_single_first_piola_kirchhoff_stress_tensor(
    x_coordinates: Tensor, x_parameters: Tensor
) -> Tensor:
    ### Plane strain assumed
    # Deformation gradient
    F_2D = calculate_single_deformation_gradient(x_coordinates)
    F = torch.stack(
        (
            torch.concat((F_2D[0, :], torch.tensor([0.0])), dim=0),
            torch.concat((F_2D[1, :], torch.tensor([0.0])), dim=0),
            torch.tensor([0.0, 0.0, 1.0]),
        ),
        dim=0,
    )
    F_transpose = torch.transpose(F, 0, 1)

    # Right Cauchy-Green tensor
    C = torch.matmul(F_transpose, F)

    # Invariants
    J = torch.unsqueeze(torch.det(F), dim=0)
    I_C = torch.trace(C)  # First invariant

    # Material parameters
    param_K = x_parameters[0]
    param_G = x_parameters[1]

    # 2. Piola-Kirchoff stress tensor
    I = torch.eye(3)
    C_inverse = torch.inverse(C)
    T_vol = param_K / 2 * (J**2 - 1) * C_inverse
    T_iso = param_G * (J ** (-2 / 3)) * (I - (1 / 3) * I_C * C_inverse)
    T = T_vol + T_iso

    # 1. Piola-Kirchoff stress tensor
    P = torch.matmul(F, T)
    P_2D = P[0:2, 0:2]
    return P_2D


def _calculate_single_cauchy_stress_tensor(
    x_coordinates: Tensor, x_parameters: Tensor
) -> Tensor:
    P = _calculate_single_first_piola_kirchhoff_stress_tensor(
        x_coordinates, x_parameters
    )
    F = calculate_single_deformation_gradient(x_coordinates)
    F_transpose = torch.transpose(F, 0, 1)
    J = torch.unsqueeze(torch.det(F), dim=0)
    return J ** (-1) * torch.matmul(P, F_transpose)


def _calculate_single_traction(
    x_coordinates: Tensor, x_parameters: Tensor, normal_vector: Tensor
) -> Tensor:
    stress = _calculate_single_first_piola_kirchhoff_stress_tensor(
        x_coordinates, x_parameters
    )
    return torch.matmul(stress, normal_vector)


@pytest.mark.parametrize(
    ("x_coordinates"),
    [
        torch.tensor([2.0, 2.0], requires_grad=True),
        torch.tensor([0.0, 0.0], requires_grad=True),
        torch.tensor([-2.0, -2.0], requires_grad=True),
    ],
)
def test_deformation_gradient_func(
    transformed_fake_ansatz: Module,
    x_coordinates: Tensor,
) -> None:
    x_parameters = torch.tensor([0.0, 0.0], requires_grad=True)

    sut = _deformation_gradient_func

    actual = sut(transformed_fake_ansatz, x_coordinates, x_parameters)

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
            torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        ),
        (
            torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        ),
    ],
)
def test_first_piola_stress_tensor_func(
    fake_ansatz: Module, x_coordinates: Tensor, x_parameters: Tensor
) -> None:
    sut = first_piola_kirchhoff_stress_func_factory()

    actual = sut(fake_ansatz, x_coordinates, x_parameters)

    expected = calculate_first_piola_kirchhof_stress(x_coordinates, x_parameters)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_coordinates", "x_parameters"),
    [
        (
            torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        ),
        (
            torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        ),
    ],
)
def test_cauchy_stress_tensor_func(
    fake_ansatz: Module, x_coordinates: Tensor, x_parameters: Tensor
) -> None:
    sut = cauchy_stress_func_factory()

    actual = sut(fake_ansatz, x_coordinates, x_parameters)

    expected = calculate_cauchy_stress(x_coordinates, x_parameters)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_coordinates", "x_parameters", "normal_vectors"),
    [
        (
            torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
            torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
    ],
)
def test_traction_func(
    fake_ansatz: Module,
    x_coordinates: Tensor,
    x_parameters: Tensor,
    normal_vectors: Tensor,
) -> None:
    sut = traction_func_factory()

    actual = sut(fake_ansatz, x_coordinates, x_parameters, normal_vectors)

    expected = calculate_traction(x_coordinates, x_parameters, normal_vectors)
    torch.testing.assert_close(actual, expected)
