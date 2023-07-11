from collections import namedtuple

import numpy as np

from parametricpinn.types import NPArray

MomentsUnivariateNormal = namedtuple(
    "MomentsUnivariateNormal", ["mean", "standard_deviation"]
)


def _determine_moments_of_univariate_normal_distribution(
    samples: NPArray,
) -> MomentsUnivariateNormal:
    mean = np.mean(samples, axis=0)
    standard_deviation = np.std(samples, axis=0, ddof=1)
    return MomentsUnivariateNormal(mean=mean, standard_deviation=standard_deviation)


MomentsMultivariateNormal = namedtuple(
    "MomentsMultivariateNormal", ["mean", "covariance"]
)


def _determine_moments_of_multivariate_normal_distribution(
    samples: NPArray,
) -> MomentsMultivariateNormal:
    mean = np.mean(samples, axis=0)
    covariance = np.cov(samples, rowvar=False)
    return MomentsMultivariateNormal(mean=mean, covariance=covariance)
