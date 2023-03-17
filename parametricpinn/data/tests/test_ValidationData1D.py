# Standard library imports

# Third-party imports
import pytest
import torch

# Local library imports
from parametricpinn.data import (
    calculate_displacements_solution_1D,
    ValidationDataset1D,
    collate_validation_data_1D,
)
from parametricpinn.settings import set_seed
from parametricpinn.types import Tensor

random_seed = 0


# def calculate_displacements(
#     coordinates: Tensor,
#     length: float,
#     youngs_modulus: Tensor,
#     traction: float,
#     volume_force: float,
# ):
#     return (traction / youngs_modulus) * coordinates + (
#         volume_force / youngs_modulus
#     ) * (length * coordinates - 1 / 2 * coordinates**2)


@pytest.mark.parametrize(
    ("coordinate", "expected"),
    [
        (torch.tensor([[0.0]]), torch.tensor([[0.0]])),
        (torch.tensor([[1.0]]), torch.tensor([[10.0]])),
        (torch.tensor([[2.0]]), torch.tensor([[18.0]])),
    ],
)
def test_calculate_displacements_solution_1D(
    coordinate: Tensor, expected: Tensor
) -> None:
    sut = calculate_displacements_solution_1D
    length = 4.0
    youngs_modulus = 1.0
    traction = 3.0
    volume_force = 2.0

    actual = sut(
        coordinates=coordinate,
        length=length,
        youngs_modulus=youngs_modulus,
        traction=traction,
        volume_force=volume_force,
    )
    torch.testing.assert_close(actual, expected)


class TestValidationDataset1D:
    length = 10.0
    traction = 1.0
    volume_force = 2.0
    min_youngs_modulus = 3.0
    max_youngs_modulus = 4.0
    num_points = 3
    num_samples = 3

    @pytest.fixture
    def sut(self) -> ValidationDataset1D:
        set_seed(random_seed)
        return ValidationDataset1D(
            length=self.length,
            traction=self.traction,
            volume_force=self.volume_force,
            min_youngs_modulus=self.min_youngs_modulus,
            max_youngs_modulus=self.max_youngs_modulus,
            num_points=self.num_points,
            num_samples=self.num_samples,
        )

    @pytest.fixture
    def x_coordinates_and_youngs_modulus_list(self) -> tuple[list[Tensor], list[float]]:
        # The random numbers must be generated in the same order as in the system under test.
        set_seed(random_seed)
        coordinates_array = (
            torch.rand((self.num_points, self.num_samples)) * self.length
        )
        coordinates_list = torch.chunk(coordinates_array, self.num_samples, dim=1)
        youngs_modulus_list = (
            self.min_youngs_modulus
            + torch.rand((self.num_samples))
            * (self.max_youngs_modulus - self.min_youngs_modulus)
        ).tolist()

        return coordinates_list, youngs_modulus_list

    def test_len(self, sut: ValidationDataset1D) -> None:
        actual = len(sut)

        expected = self.num_samples
        assert actual == expected

    @pytest.mark.parametrize(("idx"), range(num_samples))
    def test_input_sample(
        self,
        sut: ValidationDataset1D,
        x_coordinates_and_youngs_modulus_list: tuple[list[Tensor], list[float]],
        idx: int,
    ) -> None:
        actual, _ = sut[idx]

        (
            x_coordinates_list,
            x_youngs_modulus_list,
        ) = x_coordinates_and_youngs_modulus_list
        x_coordinates = x_coordinates_list[idx]
        x_youngs_modulus = torch.full((self.num_points, 1), x_youngs_modulus_list[idx])
        expected = torch.concat((x_coordinates, x_youngs_modulus), dim=1)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(("idx"), range(num_samples))
    def test_output_sample(self, sut: ValidationDataset1D, idx: int) -> None:
        input, actual = sut[idx]
        x_coordinates = input[:, 0].view((self.num_points, 1))
        x_youngs_modulus = input[:, 1].view((self.num_points, 1))

        expected = calculate_displacements_solution_1D(
            coordinates=x_coordinates,
            length=self.length,
            youngs_modulus=x_youngs_modulus,
            traction=self.traction,
            volume_force=self.volume_force,
        )
        torch.testing.assert_close(actual, expected)


class TestCollateValidationData1D:
    @pytest.fixture
    def fake_batch(self) -> list[tuple[Tensor, Tensor]]:
        sample_x_0 = torch.tensor([[1.0, 1.1]])
        sample_y_true_0 = torch.tensor([[2.0]])
        sample_x_1 = torch.tensor([[10.0, 10.1]])
        sample_y_true_1 = torch.tensor([[20.0]])
        return [(sample_x_0, sample_y_true_0), (sample_x_1, sample_y_true_1)]

    def test_batch_pde__x(self, fake_batch: list[tuple[Tensor, Tensor]]):
        sut = collate_validation_data_1D

        actual, _ = sut(fake_batch)

        expected = torch.tensor([[1.0, 1.1], [10.0, 10.1]])
        torch.testing.assert_close(actual, expected)

    def test_batch_pde__y_true(self, fake_batch: list[tuple[Tensor, Tensor]]):
        sut = collate_validation_data_1D

        _, actual = sut(fake_batch)

        expected = torch.tensor([[2.0], [20.0]])
        torch.testing.assert_close(actual, expected)
