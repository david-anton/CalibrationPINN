import numpy as np

from parametricpinn.types import NPArray


def assert_numpy_arrays_equal(actual: NPArray, expected: NPArray) -> None:
    assert actual.shape == expected.shape
    np.testing.assert_array_almost_equal(actual, expected)