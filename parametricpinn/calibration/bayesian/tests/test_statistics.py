import numpy as np
import pytest

from parametricpinn.calibration.bayesian.statistics import (
    MomentsMultivariateNormal,
    MomentsUnivariateNormal,
    determine_moments_of_multivariate_normal_distribution,
    determine_moments_of_univariate_normal_distribution,
)
from parametricpinn.tests.asserts import assert_numpy_arrays_equal


def test_determine_moments_of_univariate_normal_distribution_for_1D_samples():
    scale = 1.0
    center = 0.0
    samples = np.array([-scale, center, scale])
    num_samples = samples.shape[0]

    actual = determine_moments_of_univariate_normal_distribution(samples)

    mean = (-scale + center + scale) / num_samples
    standard_deviation = np.sqrt(
        ((-scale - mean) ** 2 + (center - mean) ** 2 + (scale - mean) ** 2)
        / (num_samples - 1)
    )
    expected = MomentsUnivariateNormal(mean=mean, standard_deviation=standard_deviation)
    assert actual == pytest.approx(expected)


def test_determine_moments_of_univariate_normal_distribution_for_2D_samples():
    scale = 1.0
    center = 0.0
    samples = np.array([[-scale], [0.0], [scale]])
    num_samples = samples.shape[0]

    actual = determine_moments_of_univariate_normal_distribution(samples)

    mean = (-scale + center + scale) / num_samples
    standard_deviation = np.sqrt(
        ((-scale - mean) ** 2 + (center - mean) ** 2 + (scale - mean) ** 2)
        / (num_samples - 1)
    )
    expected = MomentsUnivariateNormal(mean=mean, standard_deviation=standard_deviation)
    assert actual == pytest.approx(expected)


def test_determine_moments_of_multivariate_normal_distribution_mean():
    scale_a = 1.0
    center_a = 0.0
    scale_b = 2.0
    center_b = 0.0
    samples = np.array([[-scale_a, -scale_b], [center_a, center_b], [scale_a, scale_b]])
    num_samples = samples.shape[0]

    moments = determine_moments_of_multivariate_normal_distribution(samples)
    actual = moments.mean

    mean_a = (-scale_a + center_a + scale_a) / num_samples
    mean_b = (-scale_b + center_b + scale_b) / num_samples
    expected = np.array([mean_a, mean_b])
    assert_numpy_arrays_equal(actual, expected)


def test_determine_moments_of_multivariate_normal_distribution_covariance():
    scale_a = 1.0
    center_a = 0.0
    scale_b = 2.0
    center_b = 0.0
    samples = np.array([[-scale_a, -scale_b], [center_a, center_b], [scale_a, scale_b]])
    num_samples = samples.shape[0]

    moments = determine_moments_of_multivariate_normal_distribution(samples)
    actual = moments.covariance

    mean_a = (-scale_a + center_a + scale_a) / num_samples
    mean_b = (-scale_b + center_b + scale_b) / num_samples
    variance_a = (
        (-scale_a - mean_a) ** 2 + (center_a - mean_a) ** 2 + (scale_a - mean_a) ** 2
    ) / (num_samples - 1)
    variance_b = (
        (-scale_b - mean_b) ** 2 + (center_b - mean_b) ** 2 + (scale_b - mean_b) ** 2
    ) / (num_samples - 1)
    covariance_ab = (
        (-scale_a - mean_a) * (-scale_b - mean_b)
        + (center_a - mean_a) * (center_b - mean_b)
        + (scale_a - mean_a) * (scale_b - mean_b)
    ) / (num_samples - 1)
    expected = np.array([[variance_a, covariance_ab], [covariance_ab, variance_b]])
    assert_numpy_arrays_equal(actual, expected)


def test_determine_moments_of_multivariate_normal_distribution_for_univariate_distribution_mean():
    scale = 1.0
    center = 0.0
    samples = np.array([[-scale], [center], [scale]])
    num_samples = samples.shape[0]

    moments = determine_moments_of_multivariate_normal_distribution(samples)
    actual = moments.mean

    mean = (-scale + center + scale) / num_samples
    expected = np.array([mean])
    assert_numpy_arrays_equal(actual, expected)


def test_determine_moments_of_multivariate_normal_distribution_for_univariate_distribution_covariance():
    scale = 1.0
    center = 0.0
    samples = np.array([[-scale], [center], [scale]])
    num_samples = samples.shape[0]

    moments = determine_moments_of_multivariate_normal_distribution(samples)
    actual = moments.covariance

    mean = (-scale + center + scale) / num_samples

    variance = ((-scale - mean) ** 2 + (center - mean) ** 2 + (scale - mean) ** 2) / (
        num_samples - 1
    )
    expected = np.array([variance])
    print(actual)
    np.testing.assert_array_almost_equal(actual, expected)
    assert_numpy_arrays_equal(actual, expected)
