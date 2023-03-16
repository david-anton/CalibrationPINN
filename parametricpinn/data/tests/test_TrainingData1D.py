# Standard library imports

# Third-party imports
import pytest

# Local library imports
from parametricpinn.data import (
    TrainingData1D,
    TrainingDataset1D,
    collate_training_data_1D,
)


class TestTrainingDataset1D:
    length = 10.0
    traction = 2.0
    min_youngs_modulus = 3.0
    max_youngs_modulus = 4.0
    num_points_pde = 3
    num_samples = 2

    @pytest.fixture
    def sut(self) -> TrainingDataset1D:
        return TrainingDataset1D(
            length=self.length,
            traction=self.traction,
            min_youngs_modulus=self.min_youngs_modulus,
            max_youngs_modulus=self.max_youngs_modulus,
            num_points_pde=self.num_points_pde,
            num_samples=self.num_samples,
        )

    def test_len(self, sut: TrainingDataset1D) -> None:
        actual = len(sut)

        expected = self.num_samples
        assert actual == expected

    def test_first_sample_pde(self, sut: TrainingDataset1D) -> None:
        actual, _ = sut[0]

        expected = 
