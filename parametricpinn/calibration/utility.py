from collections import namedtuple

import numpy as np
import scipy.stats

from parametricpinn.types import Module, NPArray


def _freeze_model(model: Module) -> None:
    model.train(False)
    for parameters in model.parameters():
        parameters.requires_grad = False


MomentsMultivariateNormal = namedtuple(
    "MomentsMultivariateNormal", ["mean", "covariance"]
)


def _determine_moments_of_multivariate_normal_distribution(
    samples: NPArray,
) -> MomentsMultivariateNormal:
    mean = np.mean(samples, axis=0)
    covariance = np.cov(samples.T)
    return MomentsMultivariateNormal(mean=mean, covariance=covariance)


MomentsUnivariateNormal = namedtuple(
    "MomentsUnivariateNormal", ["mean", "standard_deviation"]
)


def _determine_moments_of_univariate_normal_distribution(
    samples: NPArray,
) -> MomentsUnivariateNormal:
    mean = np.mean(samples, axis=0)
    standard_deviation = np.std(samples, axis=0)
    return MomentsUnivariateNormal(mean=mean, standard_deviation=standard_deviation)
