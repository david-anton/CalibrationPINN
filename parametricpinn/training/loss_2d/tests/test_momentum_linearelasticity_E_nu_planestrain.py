import pytest
import torch

from parametricpinn.training.loss_2d.momentum_linearelasticity_E_nu import (
    momentum_equation_func_factory,
    strain_energy_func_factory,
    stress_func_factory,
    traction_energy_func_factory,
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


def calculate_stress(x_coordinates: Tensor, x_parameters: Tensor) -> Tensor:
    return torch.stack(
        [
            _calculate_single_stress_tensor(x_coordinate, x_parameter)
            for (x_coordinate, x_parameter) in zip(
                torch.unbind(x_coordinates),
                torch.unbind(x_parameters),
            )
        ]
    )


def calculate_traction(
    x_coordinates: Tensor, x_parameters: Tensor, normal_vectors: Tensor
) -> Tensor:
    return torch.stack(
        [
            _calculate_single_traction(x_coordinate, x_parameter, normal_vector)
            for (x_coordinate, x_parameter, normal_vector) in zip(
                torch.unbind(x_coordinates),
                torch.unbind(x_parameters),
                torch.unbind(normal_vectors),
            )
        ]
    )


def calculate_strain_energy(
    x_coordinates: Tensor, x_parameters: Tensor, area: Tensor
) -> Tensor:
    strain_energies = torch.stack(
        [
            _calculate_single_strain_energy(x_coordinate, x_parameter)
            for (x_coordinate, x_parameter) in zip(
                torch.unbind(x_coordinates),
                torch.unbind(x_parameters),
            )
        ]
    )
    num_collocation_points = x_coordinates.size(dim=0)
    return 1 / 2 * (area / num_collocation_points) * torch.sum(strain_energies)


def calculate_traction_energy(
    x_coordinates: Tensor,
    x_parameters: Tensor,
    normal_vectors: Tensor,
    area_fractions: Tensor,
) -> Tensor:
    traction_energies = torch.stack(
        [
            _calculate_single_traction_energy(x_coordinate, x_parameter, normal_vector)
            for (x_coordinate, x_parameter, normal_vector) in zip(
                torch.unbind(x_coordinates),
                torch.unbind(x_parameters),
                torch.unbind(normal_vectors),
            )
        ]
    )
    traction_energy_fraction = traction_energies * area_fractions
    return 1 / 2 * torch.sum(traction_energy_fraction)


def _calculate_single_stress_tensor(
    x_coordinate: Tensor, x_parameter: Tensor
) -> Tensor:
    strain = _calculate_single_strain_tensor(x_coordinate)
    voigt_strain = torch.tensor([strain[0, 0], strain[1, 1], 2 * strain[0, 1]])
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
    return torch.tensor(
        [[voigt_stress[0], voigt_stress[2]], [voigt_stress[2], voigt_stress[1]]]
    )


def _calculate_single_traction(
    x_coordinate: Tensor, x_parameter: Tensor, normal_vector: Tensor
) -> Tensor:
    stress = _calculate_single_stress_tensor(x_coordinate, x_parameter)
    return torch.matmul(stress, normal_vector)


def _calculate_single_strain_energy(
    x_coordinate: Tensor, x_parameter: Tensor
) -> Tensor:
    stress = _calculate_single_stress_tensor(x_coordinate, x_parameter)
    strain = _calculate_single_strain_tensor(x_coordinate)
    return torch.unsqueeze(torch.einsum("ij,ij", stress, strain), dim=0)


def _calculate_single_traction_energy(
    x_coordinate: Tensor, x_parameter: Tensor, normal_vector: Tensor
) -> Tensor:
    traction = _calculate_single_traction(x_coordinate, x_parameter, normal_vector)
    constants_dislpacement = torch.tensor(
        [constant_displacement_x, constant_displacement_y]
    )
    displacement = constants_dislpacement * torch.tensor(
        [
            (x_coordinate[0] ** 2) * x_coordinate[1],
            x_coordinate[0] * (x_coordinate[1] ** 2),
        ]
    )
    return torch.unsqueeze(torch.einsum("i,i", traction, displacement), dim=0)


def _calculate_single_strain_tensor(x_coordinate: Tensor) -> Tensor:
    strain_xx = (
        1 / 2 * (2 * (constant_displacement_x * 2 * x_coordinate[0] * x_coordinate[1]))
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
        1 / 2 * (2 * (constant_displacement_y * x_coordinate[0] * 2 * x_coordinate[1]))
    )
    return torch.tensor([[strain_xx, strain_xy], [strain_xy, strain_yy]])


@pytest.mark.parametrize(
    ("x_coordinates", "expected"),
    [
        (
            torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        ),
        (
            torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        ),
        (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        ),
    ],
)
def test_momentum_equation_func(
    fake_ansatz: Module,
    x_coordinates: Tensor,
    expected: Tensor,
):
    volume_forces = generate_volume_force(x_coordinates)
    x_parameters = torch.tensor(
        [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
    )
    sut = momentum_equation_func_factory(model=model)

    actual = sut(fake_ansatz, x_coordinates, x_parameters, volume_forces)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "x_coordinates",
    [
        torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True),
        torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
        torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True),
    ],
)
def test_stress_func(
    fake_ansatz: Module,
    x_coordinates: Tensor,
):
    x_parameters = torch.tensor(
        [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
    )
    sut = stress_func_factory(model=model)

    actual = sut(fake_ansatz, x_coordinates, x_parameters)

    expected = calculate_stress(x_coordinates, x_parameters)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("x_coordinates", "normal_vectors"),
    [
        (
            torch.tensor([[-2.0, -2.0], [-2.0, -2.0]], requires_grad=True),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            torch.tensor([[-2.0, 0.0], [-2.0, 0.0]], requires_grad=True),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
        (
            torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], requires_grad=True),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ),
    ],
)
def test_traction_func(
    fake_ansatz: Module,
    x_coordinates: Tensor,
    normal_vectors: Tensor,
):
    x_parameters = torch.tensor(
        [[youngs_modulus, poissons_ratio], [youngs_modulus, poissons_ratio]]
    )
    sut = traction_func_factory(model=model)

    actual = sut(fake_ansatz, x_coordinates, x_parameters, normal_vectors)

    expected = calculate_traction(x_coordinates, x_parameters, normal_vectors)
    torch.testing.assert_close(actual, expected)


def test_strain_energy_func(fake_ansatz: Module):
    x_coordinates = torch.tensor(
        [[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]], requires_grad=True
    )
    x_parameters = torch.tensor(
        [
            [youngs_modulus, poissons_ratio],
            [youngs_modulus, poissons_ratio],
            [youngs_modulus, poissons_ratio],
        ]
    )
    area = torch.tensor(16.0)
    sut = strain_energy_func_factory(model=model)

    actual = sut(fake_ansatz, x_coordinates, x_parameters, area)

    expected = calculate_strain_energy(x_coordinates, x_parameters, area)
    torch.testing.assert_close(actual, expected)


def test_traction_energy_func(fake_ansatz: Module):
    x_coordinates = torch.tensor(
        [[-2.0, -2.0], [-2.0, 0.0], [-2.0, 2.0]], requires_grad=True
    )
    x_parameters = torch.tensor(
        [
            [youngs_modulus, poissons_ratio],
            [youngs_modulus, poissons_ratio],
            [youngs_modulus, poissons_ratio],
        ]
    )
    normal_vectors = torch.tensor([[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    area_fractions = torch.tensor([[1 / 3 * 4], [1 / 3 * 4], [1 / 3 * 4]])
    sut = traction_energy_func_factory(model=model)

    actual = sut(
        fake_ansatz, x_coordinates, x_parameters, normal_vectors, area_fractions
    )

    expected = calculate_traction_energy(
        x_coordinates, x_parameters, normal_vectors, area_fractions
    )
    torch.testing.assert_close(actual, expected)
