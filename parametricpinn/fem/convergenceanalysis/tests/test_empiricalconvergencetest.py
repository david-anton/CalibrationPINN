import pytest

from parametricpinn.fem.convergenceanalysis.empiricalconvergencetest import (
    calculate_empirical_convegrence_order,
)


@pytest.mark.parametrize(
    ("results", "reduction_factor", "expected"),
    [
        ([4.0, 2.0, 1.0], 1/2, 1.0),
        ([16.0, 4.0, 1.0], 1/2, 2.0),
        ([64.0, 8.0, 1.0], 1/2, 3.0),
        ([9.0, 3.0, 1.0], 1/3, 1.0),
        ([81.0, 9.0, 1.0], 1/3, 2.0),
        ([729.0, 27.0, 1.0], 1/3, 3.0),
    ],
)
def test_calculate_empirical_convergence_order(results: list[float], reduction_factor: float, expected: float) -> None:
    sut = calculate_empirical_convegrence_order

    actual = sut(results=results, reduction_factor=reduction_factor)

    assert actual == pytest.approx(expected)