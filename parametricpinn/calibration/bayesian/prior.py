import torch

from parametricpinn.types import (
    Device,
    Tensor,
    TorchMultiNormalDist,
    TorchUniNormalDist,
)


def compile_univariate_normal_prior(
    mean: float, standard_deviation: float, device: Device
) -> TorchUniNormalDist:
    return torch.distributions.Normal(
        loc=torch.tensor(mean, dtype=torch.float64, device=device),
        scale=torch.tensor(standard_deviation, dtype=torch.float64, device=device),
    )


def compile_multivariate_normal_prior(
    means: Tensor, covariance_matrix: Tensor, device: Device
) -> TorchMultiNormalDist:
    return torch.distributions.MultivariateNormal(
        loc=means.type(torch.float64).to(device),
        covariance_matrix=covariance_matrix.type(torch.float64).to(device),
    )
