import pytest
import torch
import torch.nn as nn

from parametricpinn.training.loss_2d import (
    momentum_equation_func_factory,
    traction_func_factory,
)
from parametricpinn.training.loss_2d.tests.testdoubles import FakeAnsatz
from parametricpinn.types import Module, Tensor

model = "plane strain"
youngs_modulus = 1.0
poissons_ratio = 0.25
shear_modulus = youngs_modulus / (2 * (1 + poissons_ratio))
constant_displacement_x = 1 / 10
constant_displacement_y = -2 / 5


@pytest.fixture
def fake_ansatz() -> Module:
    return FakeAnsatz(constant_displacement_x, constant_displacement_y)


def generate_volume_force(x_coordinates: Tensor) -> Tensor:
    return torch.concat(
        (
            shear_modulus * torch.unsqueeze(x_coordinates[:, 1], dim=1),
            2 * shear_modulus * torch.unsqueeze(x_coordinates[:, 0], dim=1),
        ),
        dim=1,
    )


def calculate_traction(
    x_coordinates: Tensor, x_parameters: Tensor, normal_vectors: Tensor
) -> Tensor:
    def calculate_single_traction(
        x_coordinate: Tensor, x_parameter: Tensor, normal_vector: Tensor
    ) -> Tensor:
        strain_xx = (
            1
            / 2
            * (2 * (constant_displacement_x * 2 * x_coordinate[0] * x_coordinate[1]))
        )
        strain_xy = (
            1
            / 2
            * (
                (constant_displacement_x * x_coordinate[0] ** 2)
                + (constant_displacement_y * x_coordinate[1] ** 2)
            )
        )
        strain_yy = (
            1
            / 2
            * (2 * (constant_displacement_y * x_coordinate[0] * 2 * x_coordinate[1]))
        )
        voigt_strain = torch.tensor([strain_xx, strain_yy, 2 * strain_xy])
        E = x_parameter[0]
        nu = x_parameter[1]
        material_tensor = (E / ((1.0 + nu) * (1.0 - 2 * nu))) * torch.tensor(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2 * nu) / 2.0],
            ]
        )
        voigt_stress = torch.matmul(material_tensor, voigt_strain)
        stress = torch.tensor(
            [[voigt_stress[0], voigt_stress[2]], [voigt_stress[2], voigt_stress[1]]]
        )
        return torch.matmul(stress, normal_vector)

    return torch.stack(
        [
            calculate_single_traction(x_coordinate, x_parameter, normal_vector)
            for (x_coordinate, x_parameter, normal_vector) in zip(
                torch.unbind(x_coordinates),
                torch.unbind(x_parameters),
                torch.unbind(normal_vectors),
            )
        ]
    )


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
    volume_forces = generate_volume_force(x_coordinates)
    sut = momentum_equation_func_factory(model=model)

    actual = sut(fake_ansatz, x_coordinates, x_parameters, volume_forces)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_coordinates", "x_parameters", "normal_vectors"),
    [
        (
            torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True),
            torch.tensor(
                [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
            ),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
            torch.tensor(
                [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
            ),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True),
            torch.tensor(
                [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
            ),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
    ],
)
def test_traction_func(
    fake_ansatz: Module,
    x_coordinates: Tensor,
    x_parameters: Tensor,
    normal_vectors: Tensor,
):
    sut = traction_func_factory(model=model)

    actual = sut(fake_ansatz, x_coordinates, x_parameters, normal_vectors)

    expected = calculate_traction(x_coordinates, x_parameters, normal_vectors)
    torch.testing.assert_close(actual, expected)
