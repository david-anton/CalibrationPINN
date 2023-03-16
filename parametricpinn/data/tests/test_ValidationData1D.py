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


class TestValidationDataset1D:
    length = 10.0
    traction = 1.0
    volume_force = 2.0
    min_youngs_modulus = 3.0
    max_youngs_modulus = 4.0
    num_points = 4
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
