# Standard library imports

# Third-party imports
import pytest
import torch

# Local library imports
from parametricpinn.data import (
    TrainingData1D,
    TrainingDataset1D,
    collate_training_data_1D,
)
from parametricpinn.types import Tensor


class TestTrainingDataset1D:
    length = 10.0
    traction = 1.0
    min_youngs_modulus = 2.0
    max_youngs_modulus = 3.0
    num_points_pde = 3
    num_points_stress_bc = 1
    num_samples = 3

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

    @pytest.mark.parametrize(("idx"), (0, 1, 2))
    def test_sample_pde__x_coordinate(self, sut: TrainingDataset1D, idx: int) -> None:
        sample_pde, _ = sut[idx]

        actual = sample_pde.x_coor

        expected = torch.linspace(0.0, self.length, self.num_points_pde).view(
            (self.num_points_pde, 1)
        )
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        ("idx", "expected"),
        [
            (0, torch.full((num_points_pde, 1), min_youngs_modulus)),
            (
                1,
                torch.full(
                    (num_points_pde, 1),
                    (min_youngs_modulus + max_youngs_modulus) / 2,
                ),
            ),
            (2, torch.full((num_points_pde, 1), max_youngs_modulus)),
        ],
    )
    def test_sample_pde__x_youngs_modulus(
        self, sut: TrainingDataset1D, idx: int, expected: Tensor
    ) -> None:
        sample_pde, _ = sut[idx]

        actual = sample_pde.x_E

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(("idx"), (0, 1, 2))
    def test_sample_pde__y_true(self, sut: TrainingDataset1D, idx: int) -> None:
        sample_pde, _ = sut[idx]

        actual = sample_pde.y_true

        expected = torch.zeros((self.num_points_pde, 1))
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(("idx"), (0, 1, 2))
    def test_sample_stress_bc__x_coordinate(
        self, sut: TrainingDataset1D, idx: int
    ) -> None:
        _, sample_stress_bc = sut[idx]

        actual = sample_stress_bc.x_coor

        expected = torch.full((self.num_points_stress_bc, 1), self.length)
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        ("idx", "expected"),
        [
            (0, torch.full((num_points_stress_bc, 1), min_youngs_modulus)),
            (
                1,
                torch.full(
                    (num_points_stress_bc, 1),
                    (min_youngs_modulus + max_youngs_modulus) / 2,
                ),
            ),
            (2, torch.full((num_points_stress_bc, 1), max_youngs_modulus)),
        ],
    )
    def test_sample_stress_bc__x_youngs_modulus(
        self, sut: TrainingDataset1D, idx: int, expected: Tensor
    ) -> None:
        _, sample_stress_bc = sut[idx]

        actual = sample_stress_bc.x_E

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(("idx"), (0, 1, 2))
    def test_sample_stress_bc__y_true(self, sut: TrainingDataset1D, idx: int) -> None:
        _, sample_stress_bc = sut[idx]

        actual = sample_stress_bc.y_true

        expected = torch.full((self.num_points_stress_bc, 1), self.traction)
        torch.testing.assert_close(actual, expected)


class TestCollateTrainingData1D:
    @pytest.fixture
    def batch(self) -> list[tuple[TrainingData1D, TrainingData1D]]:
        sample_pde_0 = TrainingData1D(
            x_coor=torch.Tensor([[1.0]]),
            x_E=torch.Tensor([[1.1]]),
            y_true=torch.Tensor([[1.2]]),
        )
        sample_stress_bc_0 = TrainingData1D(
            x_coor=torch.Tensor([[2.0]]),
            x_E=torch.Tensor([[2.1]]),
            y_true=torch.Tensor([[2.2]]),
        )
        sample_pde_1 = TrainingData1D(
            x_coor=torch.Tensor([[10.0]]),
            x_E=torch.Tensor([[10.1]]),
            y_true=torch.Tensor([[10.2]]),
        )
        sample_stress_bc_1 = TrainingData1D(
            x_coor=torch.Tensor([[20.0]]),
            x_E=torch.Tensor([[20.1]]),
            y_true=torch.Tensor([[20.2]]),
        )
        return [(sample_pde_0, sample_stress_bc_0), (sample_pde_1, sample_stress_bc_1)]

    def test_batch_pde__x_coordinate(
        self, batch: list[tuple[TrainingData1D, TrainingData1D]]
    ):
        sut = collate_training_data_1D

        batch_pde, _ = sut(batch)
        actual = batch_pde.x_coor

        expected = torch.Tensor([[1.0], [10.0]])
        torch.testing.assert_close(actual, expected)

    def test_batch_pde__x_youngs_modulus(
        self, batch: list[tuple[TrainingData1D, TrainingData1D]]
    ):
        sut = collate_training_data_1D

        batch_pde, _ = sut(batch)
        actual = batch_pde.x_E

        expected = torch.Tensor([[1.1], [10.1]])
        torch.testing.assert_close(actual, expected)

    def test_batch_pde__y_true(
        self, batch: list[tuple[TrainingData1D, TrainingData1D]]
    ):
        sut = collate_training_data_1D

        batch_pde, _ = sut(batch)
        actual = batch_pde.y_true

        expected = torch.Tensor([[1.2], [10.2]])
        torch.testing.assert_close(actual, expected)

    def test_batch_stress_bc__x_coordinate(
        self, batch: list[tuple[TrainingData1D, TrainingData1D]]
    ):
        sut = collate_training_data_1D

        _, batch_stress_bc = sut(batch)
        actual = batch_stress_bc.x_coor

        expected = torch.Tensor([[2.0], [20.0]])
        torch.testing.assert_close(actual, expected)

    def test_batch_stress_bc__x_youngs_modulus(
        self, batch: list[tuple[TrainingData1D, TrainingData1D]]
    ):
        sut = collate_training_data_1D

        _, batch_stress_bc = sut(batch)
        actual = batch_stress_bc.x_E

        expected = torch.Tensor([[2.1], [20.1]])
        torch.testing.assert_close(actual, expected)

    def test_batch_stress_bc__y_true(
        self, batch: list[tuple[TrainingData1D, TrainingData1D]]
    ):
        sut = collate_training_data_1D

        _, batch_stress_bc = sut(batch)
        actual = batch_stress_bc.y_true

        expected = torch.Tensor([[2.2], [20.2]])
        torch.testing.assert_close(actual, expected)
